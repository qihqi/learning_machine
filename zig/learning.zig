const std = @import("std");

const TimestampType = enum {
    unix,
    datetime,
};

// tagged union
const Timestamp = union(TimestampType) {
    unix: i32,
    datetime: Datetime,

    const Datetime = struct {
        year: u16,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
    };

    fn seconds(self: Timestamp) u8 {
        switch (self) {
            .unix => |ts| {
                const seconds_since_midnight: i32 = @rem(ts, 86000);
                return @intCast(@rem(seconds_since_midnight, 60));
            },
            .datetime => |ds| return ds.second,
        }
    }
};

pub fn main() void {
    const ts = Timestamp{ .unix = 1693278411 };
    const ts_p = &ts;
    const ts_pp = &ts_p;
    std.debug.print("{any} \n", .{@TypeOf(ts_pp)});
}
