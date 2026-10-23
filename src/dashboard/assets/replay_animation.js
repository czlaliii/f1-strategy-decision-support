// Smooth car animation on the replay track.
// The server sends a window of car positions (0.1 s grid) and the replay
// clock every 250 ms; here the cars are interpolated between samples and
// only the car trace (index 1) of the track figure is redrawn, ~30 fps.
(function () {
    var FRAME_MS = 33;
    // Treat clock changes larger than this many seconds (x speed) as a seek.
    var SEEK_SECONDS = 1.5;
    // Never run ahead of the last server clock by more than this many seconds (x speed).
    var MAX_LEAD_SECONDS = 0.6;

    var state = {
        anchor: null,     // last server clock: {time, at, playing, speed}
        rate: null,       // observed replay seconds per real second
        data: null,       // position window from the server
        shown: null,      // replay time currently drawn
        lastDrawn: null,
        lastDrawAt: 0
    };

    function sync(clock, paused, speed, data) {
        var now = performance.now();
        var time = Number(clock) || 0;
        var playing = !paused;
        speed = Number(speed) || 1;
        var previous = state.anchor;
        if (playing && previous && previous.playing && time > previous.time) {
            var observed = (time - previous.time) / ((now - previous.at) / 1000);
            if (observed > 0 && observed < speed * 3) {
                state.rate = state.rate === null ? observed : 0.8 * state.rate + 0.2 * observed;
            }
        }
        if (!playing || !previous || previous.speed !== speed) {
            state.rate = speed;
        }
        state.anchor = {time: time, at: now, playing: playing, speed: speed};
        state.data = data || null;
        return window.dash_clientside.no_update;
    }

    function replayTime(now) {
        var a = state.anchor;
        var target = a.time;
        if (a.playing) {
            var lead = ((now - a.at) / 1000) * (state.rate || a.speed);
            target += Math.min(lead, MAX_LEAD_SECONDS * a.speed);
        }
        var shown = state.shown;
        if (shown === null || !a.playing || Math.abs(target - shown) > SEEK_SECONDS * a.speed) {
            state.shown = target;
        } else {
            // While playing, small corrections never move the cars backwards.
            state.shown = Math.max(shown, target);
        }
        return state.shown;
    }

    function positions(d, t) {
        var pos = (t - d.t0) / d.dt;
        var last = d.x.length ? d.x[0].length - 1 : -1;
        if (pos < 0 || pos > last) {
            return null;  // waiting for the next window
        }
        var i = Math.min(Math.floor(pos), last);
        var j = Math.min(i + 1, last);
        var f = pos - i;
        var out = {x: [], y: [], text: [], color: []};
        for (var k = 0; k < d.drivers.length; k++) {
            var x0 = d.x[k][i], y0 = d.y[k][i], x1 = d.x[k][j], y1 = d.y[k][j];
            var x, y;
            if (x0 !== null && x1 !== null) {
                x = x0 + (x1 - x0) * f;
                y = y0 + (y1 - y0) * f;
            } else if (x0 !== null) {
                x = x0; y = y0;
            } else if (x1 !== null) {
                x = x1; y = y1;
            } else {
                continue;  // no sample: the car is not shown
            }
            out.x.push(x);
            out.y.push(y);
            out.text.push(d.drivers[k]);
            out.color.push(d.colors[k]);
        }
        return out;
    }

    function frame(now) {
        window.requestAnimationFrame(frame);
        if (now - state.lastDrawAt < FRAME_MS || !state.anchor || !state.data || !window.Plotly) {
            return;
        }
        var graph = document.querySelector("#replay-track .js-plotly-plot");
        if (!graph || !graph.data || graph.data.length < 2) {
            return;
        }
        var t = replayTime(now);
        // A freshly drawn figure has an empty car trace; redraw even when paused.
        var empty = !graph.data[1].x || graph.data[1].x.length === 0;
        if (t === state.lastDrawn && !empty) {
            return;
        }
        var cars = positions(state.data, t);
        if (!cars) {
            return;
        }
        state.lastDrawAt = now;
        state.lastDrawn = t;
        window.Plotly.restyle(
            graph,
            {x: [cars.x], y: [cars.y], text: [cars.text], "marker.color": [cars.color]},
            [1]
        );
    }

    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        replay: {sync: sync}
    });
    window.requestAnimationFrame(frame);
})();
