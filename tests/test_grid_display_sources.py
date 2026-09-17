"""Display must remain usable when the full global map is suspended."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'visual'))
from visual_style import GridSourceCache


class GridDisplayTests(unittest.TestCase):
    def test_global_invalidation_preserves_latest_local(self):
        cache=GridSourceCache();local={'stamp':10.,'grid':'local'}
        self.assertEqual(cache.update('local',local),('local',local))
        full={'stamp':10.,'grid':'full'}
        self.assertEqual(cache.update('global',full),('global',full))
        self.assertEqual(cache.update('global',None),('local',local))
        newer={'stamp':12.,'grid':'new local'}
        self.assertEqual(cache.update('local',newer),('local',newer))
        self.assertEqual(cache.update('global',None),('local',newer))

    def test_full_map_recovers_after_memory_pressure(self):
        cache=GridSourceCache();cache.update('local',{'stamp':8.})
        full={'stamp':9.};self.assertEqual(cache.update('global',full),('global',full))
        self.assertEqual(cache.update('local',None),('global',full))

    def test_delayed_global_does_not_hide_fresh_local(self):
        cache=GridSourceCache();cache.update('global',{'stamp':5.})
        local={'stamp':12.};self.assertEqual(cache.update('local',local),('local',local))
        self.assertEqual(cache.update('global',{'stamp':6.}),('local',local))
        full={'stamp':12.};self.assertEqual(cache.update('global',full),('global',full))

    def test_both_invalid_clear_the_map(self):
        cache=GridSourceCache();cache.update('local',{'stamp':1.})
        cache.update('global',None)
        self.assertEqual(cache.update('local',None),('',None))
        self.assertEqual(len(cache.maps),2)
        with self.assertRaises(ValueError):cache.update('other',{'stamp':1.})


if __name__=='__main__':unittest.main()
