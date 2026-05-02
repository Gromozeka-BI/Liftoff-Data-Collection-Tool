import pyarrow as pa

RC_CHANNELS_SCHEMA = pa.schema([
    pa.field("seq",          pa.int64()),
    pa.field("ts_wall",      pa.float64()),
    pa.field("ts_device_us", pa.int64()),
    pa.field("ch1",          pa.int32()),
    pa.field("ch2",          pa.int32()),
    pa.field("ch3",          pa.int32()),
    pa.field("ch4",          pa.int32()),
    pa.field("ch5",          pa.int32()),
    pa.field("ch6",          pa.int32()),
    pa.field("ch7",          pa.int32()),
    pa.field("ch8",          pa.int32()),
])

TIMELINE_SCHEMA = pa.schema([
    pa.field("seq",     pa.int64()),
    pa.field("ts_wall", pa.float64()),
])

TELEMETRY_SCHEMA = pa.schema([
    pa.field("seq",        pa.int64()),
    pa.field("ts_wall",    pa.float64()),   # unix wall clock
    pa.field("ts_sim",     pa.float32()),   # LiftOff internal timer
    pa.field("pos_x",      pa.float32()),
    pa.field("pos_y",      pa.float32()),
    pa.field("pos_z",      pa.float32()),
    pa.field("att_x",      pa.float32()),
    pa.field("att_y",      pa.float32()),
    pa.field("att_z",      pa.float32()),
    pa.field("att_w",      pa.float32()),
    pa.field("vel_x",      pa.float32()),
    pa.field("vel_y",      pa.float32()),
    pa.field("vel_z",      pa.float32()),
    pa.field("gyro_pitch", pa.float32()),
    pa.field("gyro_roll",  pa.float32()),
    pa.field("gyro_yaw",   pa.float32()),
    pa.field("in_throttle",pa.float32()),
    pa.field("in_yaw",     pa.float32()),
    pa.field("in_pitch",   pa.float32()),
    pa.field("in_roll",    pa.float32()),
    pa.field("bat_v",      pa.float32()),
    pa.field("bat_pct",    pa.float32()),
    pa.field("motor_0",    pa.float32()),
    pa.field("motor_1",    pa.float32()),
    pa.field("motor_2",    pa.float32()),
    pa.field("motor_3",    pa.float32()),
])

EVENTS_SCHEMA = pa.schema([
    pa.field("seq",        pa.int64()),
    pa.field("ts_wall",    pa.float64()),
    pa.field("event_type", pa.string()),
    pa.field("gate_id",    pa.int32()),
    pa.field("lap_num",    pa.int32()),
    pa.field("source",     pa.string()),
])
