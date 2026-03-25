const std = @import("std");
const expect = std.testing.expect;

const constant: i32 = 5;
var variable: u32 = 5000;

const inferred_constant = @as(i32, 5);
// @as is for casting
var inferred_var = @as(u32, 5);


test "always succeeds" {
    std.debug.print("{d}\n", .{variable});
    const a = [5]u8{'h', 'e', 'l', 'l', 'o'};

    std.debug.print("size is {d}\n", .{a.len});

    try expect(true);
}

test "if statement" {
    const a = true;
    var x : u16 = 0;
    if (a) {
        x += 1;
    } else {
        x += 2;
    }
    std.debug.print( "here {d} \n", .{if (a) 1 else 2});
    try expect(x == 1);
}

test "while" {
  var i: u8 = 2;
  while (i < 100) {
    i *= 2;
  }
  try expect(i == 128);
}

test "while with continue" {
  var i: u8 = 0;
  var sum: u8 = 0;
  // second term is continue expression
  // do once on continue?
  while (i <= 10) : (i += 1) {
    sum += i;
  }
  try expect(sum == 55);
}

test "while with continue 2" {
  var i: u8 = 0;
  var sum: u8 = 0;
  // second term is continue expression
  // do once on continue?
  while (i <= 3) : (i += 1) {
    std.debug.print(" inside of while {d}\n", .{i});
    if (i == 2) continue;
    sum += 1;
  }
  try expect(sum == 3);
}

// next TODO is https://zig.guide/language-basics/for-loops

test "for" {
    const string = [_]u8{'a', 'b', 'c'};
    for (string, 0..) |character, index| {
        std.debug.print("{c}: {d}\n", .{character, index});
    }
}


test "two sum" {
    var arr: [100]i32 = undefined;
    for (0..100) |i| {
        arr[i] = @mod(std.crypto.random.int(i32), 30);
    }

    std.debug.print("{any}\n", .{arr});
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const test_allocator = gpa.allocator();
    var map: std.AutoHashMap(i32, i32) = .init(test_allocator);
    defer map.deinit();

    for (arr) |content| {
        if (map.contains(content)) {
            // std.debug.print("Found: {d} + {d} = 30\n", .{content, 30-content});
        } else {
            try map.put(30 - content, content);
        }
    }

    std.debug.print("NOT found", .{});

}

fn addFive(a: u32) u32 {
    return a + 5;
}

test "function" {
    const y = addFive(12);
    std.debug.print("\ny is {d}\n", .{y});
}

test "defer" {
    var x: i16 = 5;
    {
        defer x += 2;
        try expect (x == 5);
    }
    try expect (x == 7);
}

const FileOpenError = error{
    AccessDenied,
    OutOfMemory,
    FileNotFound
};

const AllocationError = error{OutOfMemory};

test "coerce error" {
    const err: FileOpenError = AllocationError.OutOfMemory;
    try expect(err == FileOpenError.OutOfMemory);
    // the line below does not pass
    try expect(AllocationError.OutOfMemory == FileOpenError.OutOfMemory);
}

test "error union" {
    const maybe_error: AllocationError!u16 = 10;
    const no_error = maybe_error catch 0;
    try expect(@TypeOf(no_error) == u16);
    try expect(no_error == 10);
}

fn failingFunction() error{Oops}!void {
    return error.Oops;
}
fn failFn() error{Oops}!i32{
    try failingFunction();
    return 12; // unreached
}

test "returning an error" {
    // note the |err| {...} is not an lambda
    failingFunction() catch |err| {
        try expect(err == error.Oops);
        return;
    };
    // try x == try x catch |err| return err;
}

test "try" {
    const v = failFn();
    const z = v catch |err| {
        std.debug.print("try HERE {any}\n", .{err});
        return;
    };
    _ = z;
}

test "switch" {
    var x: i8 = 10;
    switch (x) {
        -1...1 => {
            x = -x;
        },
        10, 100 => {
            x = @divExact(x, 10);
        },
        else => {}
    }
    try expect(x == 1);
}

test "out of bounds" {
    @setRuntimeSafety(false);
    const a = [3]u8{1, 2, 3};
    var index: u8 = 5;
    const b = a[index];

    _ = b;
    index = index;
}

test "unreachable" {
    const x:i32 = 2;
    // type of unreachable is noreturn; no return can coerce to any type
    const y: i32 = if (x == 2) 5 else unreachable;
    _ = y;
}

fn asciiToUpper(x: u8) u8 {
    return switch(x) {
        'a' ... 'z' => x + 'A' - 'a',
        'A' ... 'Z' => x,
        else => unreachable,
    };
}

test "unreachable switch" {
    try expect(asciiToUpper('a') == 'A');
}

// pointer type
fn increment(num: *u8) void {
    // dereference syntax
    num.* += 1;
}

test "pointers" {
    var x: u8 = 1;
    // take ptr
    increment(&x);
    try expect(x == 2);
}

test "naughty pointer" {
    var x: u16 = 5;
    x -= 4;
    var y: *u8 = @ptrFromInt(x);
    y = y;
}

fn doubleAllManypointer(buffer: [*]u8, byte_count: usize) void {
  var i: usize = 0;
  while (i < byte_count) : (i += 1) {
      buffer[i] *= 2;
  }
}

test "many-item ptr" {
    var buffer: [100]u8 = [_]u8{1} ** 100;
    const buffer_ptr: *[100]u8 = &buffer;
    const buffer_many_ptr: [*]u8 = buffer_ptr;
    doubleAllManypointer(buffer_many_ptr, buffer.len);
    for (buffer) |byte| try expect(byte == 2);
}

// TODO: next is https://zig.guide/language-basics/slices

fn total(values: []const u8) usize {
    var sum: usize = 0;
    for (values) |v| sum += v;
    return sum;
}

test "slices" {
    const array = [_]u8{ 1,2,3,4,5};
    const slice = array[0..3];
    try expect(total(slice) == 6);
    try expect(@TypeOf(slice) == *const [3]u8);
}

test "slices 3" {
    const array = [_]u8{ 1,2,3,4,5};
    const slice = array[0..];
    _ = slice;
}

const Direction = enum {
    north,
    south,
    east,
    west
};

const Value = enum(u2) { zero, one, two};

test "enum ordinal" {
    try expect(@intFromEnum(Value.zero) == 0) ;
    try expect(@intFromEnum(Value.one) == 1) ;
    try expect(@intFromEnum(Value.two) == 2) ;
}

const Suit = enum {
    clubs,
    spades,
    diamonds,
    hearts,
    pub fn isClub(self: Suit) bool {
        return self == Suit.clubs;
    }
};

test "enum method" {
    try expect(Suit.spades.isClub() == Suit.isClub(.spades));
}

// next is https://zig.guide/language-basics/structs
//
const Vec3 = struct {x: f32, y: f32, z: f32 };

test "struct" {
    const my_vector : Vec3 = .{
        .x = 0,
        .y = 100,
        .z = 50,
    };
    _ = my_vector;
}


const Stuff = struct {
    x: i32,
    y: i32,
    fn swap(self: *Stuff) void {
        const tmp = self.x;
        self.x = self.y;
        self.y = tmp;
    }
};

test "auto deref" {
    var thing = Stuff{.x = 10, .y = 20};
    thing.swap();
    try expect(thing.x == 20);
}

const Result = union {
    int: i64,
    float: f64,
    bool: bool,
};

// test "simple union" {
//     var result = Result {.int = 1234};
//     result.float = 12.34;
// }
//

const Tag = enum {a, b, c};
const Tagged = union(Tag){
    a: u8,
    b: f32,
    c: bool,
};

// esto es equivalente del arriba
const Tagged2 = union(enum){
    a: u8,
    b: f32,
    c: bool,
};

test "switch union" {
    var value = Tagged{.b = 1.5};
    switch (value) {
        .a => |*byte| byte.* += 1,
        .b => |*float| float.* += 1,
        .c => |*b| b.* = !b.*,
    }
}

test "labeled blocks" {
    const count = blk: {
        var sum: u32 = 0;
        var i: u32 = 0;
        while (i < 10) : (i += 1) {
            sum += i;
        }
        break :blk sum;
    };
    _ = count;
}

test "nested countinue" {
    var count: usize = 0;
    outer: for ([_]i32{1,2,3,4,5,6,7,8}) |_| {
        for ([_]i32{1,2,3,4,5}) |_| {
            count += 1;
            continue :outer;
        }

    }
    try expect(count == 8);
}

fn rangeHasNumber(begin: usize, end: usize, number: usize) bool {
    var i = begin;
    return while (i < end) : (i += 1) {
        if (i == number) {
            break true;
        }
    } else false;
}
test  "while loop expression" {
    try expect(rangeHasNumber(0, 10, 3));
}


test "optional" {
    var found_index: ?usize = null;
    const data = [_]i32 { 1,2,3,4,5,6,7,8,12};
    for (data, 0..) |v, i| {
        if (v == 10) found_index = i;
    }
    try expect(found_index == null);
}

fn fibonacci(n: u16) u16 {
    if (n == 0 or n == 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

test "comptime" {
    const x = comptime fibonacci(10);
    const y = comptime blk: {
        break :blk fibonacci(10);
    };
    try(expect(y == 55));
    try(expect(x == 55));

}

fn Matrix (
    comptime T: type,
    comptime width: comptime_int,
    comptime height: comptime_int,
) type {
    return [height][width]T;
}

test "return atype" {
    try expect(Matrix(f32, 4, 4) == [4][4]f32);
}
// continue at https://zig.guide/language-basics/comptime unfinished
//

test "branch on types" {
    const a = 5;
    const b: if (a < 10) f32 else i32 = 5;
    try expect(b == 5);
    try expect(@TypeOf(b) == f32);
}

fn addSmallInts(comptime T: type, a: T, b: T) T {
    return switch(@typeInfo(T)) {
        .comptime_int => a + b,
        .int => |info| if (info.bits <= 16)
                a + b
            else
                @compileError("ints too large"),
        else => @compileError("only ints accepted"),
    };
}

test "typeinfo switch" {
    const x = addSmallInts(u16, 20, 30);
    try expect(@TypeOf(x) == u16);
    try expect(x == 50);
}

fn GetBiggerInt(comptime T: type) type {
    return @Type(.{
        .int = .{
            .bits = @typeInfo(T).int.bits + 1,
            .signedness = @typeInfo(T).int.signedness,
        },
    });
}

test "@Type" {
    try expect(GetBiggerInt(u8) == u9);
    try expect(GetBiggerInt(i31) == i32);
}

fn Vec(
    comptime count: comptime_int,
    comptime T: type,
) type {
    return struct {
        data: [count]T,
        const Self = @This();

        fn abs(self: Self) Self {
            var tmp = Self{ .data = undefined };
            for (self.data, 0..) |elem, i| {
                tmp.data[i] = if (elem < 0)
                    -elem
                 else
                    elem;

            }
            return tmp;
        }
        fn init(data:[count]T) Self {
            return Self{.data = data};
        }
    };
}

const eql = @import("std").mem.eql;

test "generic vector" {
    const x = Vec(3, f32).init([_]f32{10, -10, 5});
    const y = x.abs();
    try expect(eql(f32, &y.data, &[_]f32{10, 10, 5}));
}

test "optional-if" {
    const maybe_num: ?usize = 10;
    if (maybe_num) |n| {
        try expect(@TypeOf(n) == usize);
        try expect(n == 10);
    } else {
        unreachable;
    }
}

test "error union if" {
    const ent_num: error{UnknownEntity}!u32 = 5;

    if (ent_num) |entity| {
        try expect(@TypeOf(entity) == u32);
        try expect(entity == 5);
    } else |err| {
        _ = err catch {};
        unreachable;
    }
}

test "for with pointer" {
    var data = [_]u8{ 1,2,3};
    for (&data) |*bytes| bytes.* += 1;
    try expect(eql(u8, &data, &[_]u8{2, 3, 4}));
}

test "tuple" {
    const values = .{
        @as(u32, 1234),
        @as(f64, 12.34),
        true,
        "hi",
    } ++ .{false} ** 2;

    inline for(values, 0..) |v, i| {
        if (i != 2) { continue; }
        try expect(v);
    }
    try expect(values.len == 6);
    try expect(values.@"3"[0] == 'h');
}

const meta = @import("std").meta;

// VECTOR for SIMD
test "vector add" {
    const x: @Vector(4, f32) = .{ 1, -10, 20, -1};
    const y: @Vector(4, f32) = .{2, 10, 0, 1};

    const z = x + y;
    try expect(meta.eql(z, @Vector(4, f32){3, 0, 20, 0}));
}

// https://zig.guide/standard-library/allocators
test "allocation" {
   const allocator = std.heap.page_allocator;
   const memory = try allocator.alloc(u8, 100);
   defer allocator.free(memory);

   try expect(memory.len == 100);
   try expect(@TypeOf(memory) == []u8);
}

test "fixed buffer allocator" {
    var buffer: [1000]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&buffer);
    const allocator = fba.allocator();

    const memory = try allocator.alloc(u8, 100);
    defer allocator.free(memory);

    try expect(memory.len == 100);
    try expect(@TypeOf(memory) == []u8);
}

test "arena allocator" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    _ = try allocator.alloc(u8, 1);
    _ = try allocator.alloc(u8, 1);
    _ = try allocator.alloc(u8, 1);
}
