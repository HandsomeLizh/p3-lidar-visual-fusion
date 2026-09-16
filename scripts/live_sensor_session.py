#!/usr/bin/env python3
"""Compatibility entry for passive live sensor relay; vehicle control is external."""
from live_sensor_relay import main
if __name__ == '__main__':
    raise SystemExit(main())
