from dct.mavlink_geo import build_transform_from_settings


def test_track_geo_transform_maps_three_anchors() -> None:
    track = {
        "bounds": {
            "origin_x": 10.0,
            "origin_z": 20.0,
            "x": 15.0,
            "y": 30.0,
        },
    }
    settings = {
        "anchors": {
            "origin": {"lat": 55.0, "lon": 37.0, "alt": 100.0},
            "x": {"lat": 55.0001, "lon": 37.0002, "alt": 101.0},
            "z": {"lat": 55.0003, "lon": 37.0004, "alt": 102.0},
        },
    }

    transform = build_transform_from_settings(track, settings)

    assert transform is not None
    origin = transform.to_geo([10.0, 0.0, 20.0])
    x_point = transform.to_geo([25.0, 0.0, 20.0])
    z_point = transform.to_geo([10.0, 0.0, 50.0])
    high_origin = transform.to_geo([10.0, 5.0, 20.0])

    assert origin.lat == settings["anchors"]["origin"]["lat"]
    assert origin.lon == settings["anchors"]["origin"]["lon"]
    assert origin.alt == settings["anchors"]["origin"]["alt"]
    assert abs(x_point.lat - settings["anchors"]["x"]["lat"]) < 1e-9
    assert abs(x_point.lon - settings["anchors"]["x"]["lon"]) < 1e-9
    assert abs(x_point.alt - settings["anchors"]["x"]["alt"]) < 1e-9
    assert abs(z_point.lat - settings["anchors"]["z"]["lat"]) < 1e-9
    assert abs(z_point.lon - settings["anchors"]["z"]["lon"]) < 1e-9
    assert abs(z_point.alt - settings["anchors"]["z"]["alt"]) < 1e-9
    assert high_origin.alt == 105.0
