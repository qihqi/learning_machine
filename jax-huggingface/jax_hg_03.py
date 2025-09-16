import sys
import time
import jax
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import torchax as tx
from torchax.interop import torch_view

from jax.sharding import PartitionSpec as P, NamedSharding
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt")
print(model_inputs)

mesh = jax.make_mesh((jax.device_count(), ), ('axis', ))
env = tx.default_env()


# added later
from jax.tree_util import register_pytree_node
from transformers import modeling_outputs

def output_flatten(v):
  return v.to_tuple(), None

def output_unflatten(aux, children):
  return modeling_outputs.CausalLMOutputWithPast(*children)

register_pytree_node(
  modeling_outputs.CausalLMOutputWithPast,
  output_flatten,
  output_unflatten,
)

from transformers import cache_utils

def _flatten_dynamic_cache(dynamic_cache):
  return (
      dynamic_cache.key_cache,
      dynamic_cache.value_cache,
  ), None

def _unflatten_dynamic_cache(aux, children):
  cache = cache_utils.DynamicCache(),
  cache.key_cache, cache.value_cache = children
  return cache

register_pytree_node(
  cache_utils.DynamicCache,
  _flatten_dynamic_cache,
  _unflatten_dynamic_cache,
)

def _flatten_static_cache(cache):
  layers_keys = [layer.keys for layer in cache.layers]
  layers_values = [layer.values for layer in cache.layers]

  return (
      layers_keys,
      layers_values,
  ), (cache._config, cache.max_cache_len)

def _unflatten_static_cache(aux, children):
  _config, max_cache_len = aux
  cache = cache_utils.StaticCache(config=_config, max_cache_len=max_cache_len, )
  cache._config = _config
  layers_keys, layers_values = children
  for i, layer in enumerate(cache.layers):
    layer.keys = layers_keys[i]
    layer.values = layers_values[i]

  return cache

register_pytree_node(
  cache_utils.StaticCache,
  _flatten_static_cache,
  _unflatten_static_cache,
)

model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", 
        torch_dtype="bfloat16", device_map="cpu")
        
def shard_weights_llama(mesh, weights):
  result = {}
  for k, v in weights.items():
    if (('q_proj' in k) or 
       ('k_proj' in k) or 
       ('v_proj' in k) or 
       ('gate_proj' in k) or 
       ('up_proj' in k)):
      sharding = P('axis', None)
    elif(('o_proj' in k) or 
       ('down_proj' in k) or 
       ('lm_head.weight' in k) or 
       ('embed_tokens' in k)):
      sharding = P(None, 'axis')
    else:
      sharding = P() # replicated

    # apply_jax calls a jax function with the jax.Array inside of the tensor
    # as input.
    result[k] = v.apply_jax(jax.device_put, NamedSharding(mesh, sharding))
  return result 

def run_twice_and_print_cache(model, input_ids):
  res = model(input_ids)
  print('number of layers', len(res[1]))
  for k, v in res[1]:
    print('first kv cache')
    print(k.shape, v.shape)
    break

  next_token = torch.argmax(res[0][:, -1], dim=-1)

  res = model(next_token.unsqueeze(0), past_key_values=res[1])
  print('number of layers', len(res[1]))
  for k, v in res[1]:
    print('second kv cache')
    print(k.shape, v.shape)
    break


def run_twice_and_print_cache_static(model, input_ids):
  past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=50, 
                                device='jax', dtype=model.dtype)
  res = model(input_ids, past_key_values=past_key_values)
  print('number of layers', len(res[1]))
  for k, v in res[1]:
    print('first kv cache')
    print(k.shape, v.shape)
    break

  next_token = torch.argmax(res[0][:, -1], dim=-1)

  res = model(next_token.unsqueeze(0), past_key_values=res[1])
  print('number of layers', len(res[1]))
  for k, v in res[1]:
    print('second kv cache')
    print(k.shape, v.shape)
    break


def autoregressive_decode(model, input_ids, tokenizer, max_tokens=50, use_static_cache=False):
  start = time.perf_counter()
  res = model(input_ids)

  next_token = torch.argmax(res[0][:, -1], dim=-1)
  result_tokens = [int(next_token.item())]

  for _ in range(max_tokens):
    res = model(next_token.unsqueeze(0), past_key_values=res[1])
    next_token = torch.argmax(res[0][:, -1], dim=-1)
    if next_token.item() == tokenizer.eos_token:
      break
    result_tokens.append(next_token.item())
  end = time.perf_counter()
  
  print('decoded', tokenizer.batch_decode([result_tokens]))
  print(f'took {end - start} seconds')
  return result_tokens

# https://huggingface.co/docs/transformers/v4.44.0/en/llm_optims?static-kv=advanced+usage%3A+control+Static+Cache#static-kv-cache-and-torchcompile
def autoregressive_decode_static(model, input_ids, tokenizer, max_tokens=50):

  def decode_one_tokens(model_weights, cur_token, input_pos, cache_position, past_key_values):
    logits, cache = torch.func.functional_call(
        model, 
        model_weights, # weight state_dict
        (cur_token,), # args as tuple
        dict(
          position_ids=input_pos,
          cache_position=cache_position,
          past_key_values=past_key_values,
          return_dict=False,
          use_cache=True) # kwargs as dict
    )
    new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token, cache

  jitted = tx.interop.jax_jit(decode_one_tokens)
  #jitted = decode_one_tokens

  batch_size, seq_length = input_ids.shape
  model_weights = model.state_dict()
  with torch.no_grad():
    start = time.perf_counter()
    past_key_values = StaticCache(
        config=model.config, 
        max_batch_size=1, max_cache_len=max_tokens, 
        device='jax', dtype=model.dtype
    )
    past_key_values._config = model.config # keep this
    cache_position = torch.arange(seq_length, device='jax')
    generated_ids = []

    logits, past_key_values = model(
        input_ids, 
        cache_position=cache_position, 
        past_key_values=past_key_values, 
        return_dict=False, 
        use_cache=True
    )
    next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    generated_ids.append(next_token[:, 0].item())

    for k in past_key_values.key_cache:
      k.apply_jax_(jax.device_put, NamedSharding(mesh, P(None, 'axis', None, None))) # shard on num of head
    for k in past_key_values.value_cache:
      k.apply_jax_(jax.device_put, NamedSharding(mesh, P(None, 'axis', None, None))) # shard on num of head


    for layer in past_key_values.layers:
      sharding_spec = P(None, 'axis', None, None)
      layer.keys = layer.keys.apply_jax_(jax.device_put, NamedSharding(mesh, sharding_spec)) # shard on num of head
      layer.values = layer.values.apply_jax_(jax.device_put, NamedSharding(mesh, sharding_spec)) # shard on num of head
    
    cache_position = torch.tensor([seq_length + 1], device='jax')
    
    for i in range(1, max_tokens):
        iter_time = time.perf_counter()
        next_token, past_key_values = jitted(
          model_weights,
          next_token.clone(), None, cache_position, past_key_values)
        generated_ids.append(next_token.int().item())
        cache_position += 1
        #print('Iteration', i, ' took ', time.perf_counter() - iter_time)
    end = time.perf_counter()

  text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
  print(text)
  print('Time: ', end - start)


# move the model's weight to 'jax' device. i.e. a tensor with a 
# jax array inside
with env:
  model.to('jax')
  weights = shard_weights_llama(mesh, model.state_dict())
  # put the weights back into the model
  model.load_state_dict(weights, assign=True, strict=False)
  input_ids = model_inputs.input_ids.to('jax').apply_jax_(
    jax.device_put,
    NamedSharding(mesh, P()))
  tx.interop.call_jax(jax.block_until_ready, weights)

  #run_twice_and_print_cache(model, input_ids)
  #run_twice_and_print_cache_static(model, input_ids)

  autoregressive_decode_static(model, input_ids, tokenizer)


sys.exit(0)