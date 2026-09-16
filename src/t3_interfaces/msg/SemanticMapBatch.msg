# Transaction descriptor for full-tile IncrementalSemanticMap payloads.
#
# The legacy map payload remains unchanged.  BEGIN announces the exact tile
# set for one batch. COMPLETE is published only after every listed payload has
# been published and supplies the corresponding IncrementalSemanticMap
# update_id values.  Receivers buffer by session_id/batch_id and apply the
# batch only after COMPLETE and all payloads are present.

uint8 PHASE_BEGIN=0
uint8 PHASE_COMPLETE=1

std_msgs/Header header

# Stable configured map name and unique mapper-process session.
string map_id
string session_id

# Monotonic within session. map_revision is the authoritative map state;
# batch_id also advances for replays and retries of the same map revision.
uint64 map_revision
uint64 batch_id
uint8 phase

# reset is asserted until the first batch in this session completes.
# full_snapshot means tile_x/tile_y enumerate a complete frozen global map,
# rather than only changed tiles.
bool reset
bool full_snapshot

# The four tile arrays always have tile_count elements and describe complete
# authoritative 256x256 tile replacements. tile_cleared means the matching
# payload is an all-unknown tombstone that erases old receiver state.
uint32 tile_count
int32[] tile_x
int32[] tile_y
uint64[] tile_revision
bool[] tile_cleared

# Empty in BEGIN. COMPLETE contains one legacy payload update_id per tile in
# the same order as the arrays above.
uint32[] payload_update_id
