// Smooth car animation on the replay track.
// The server sends a window of car positions (0.1 s grid) and the replay
// clock every 250 ms. Here a display clock runs continuously at playback
// speed and is gently pulled towards the server clock, cars are interpolated
// between samples, and only the car trace (the last one) is redrawn each frame.
(function () {
    // Clock differences larger than this many seconds (x speed) are seeks: jump.
    var SEEK_SECONDS = 1.5;
    // Time constant (s) for pulling the display clock towards the server clock.
    var CORRECTION_SECONDS = 1.0;
    // The server clock estimate never runs further ahead of its last value (x speed).
    var MAX_LEAD_SECONDS = 1.0;
    // Redraws slower than this switch to every second frame, keeping an even rhythm.
    var SLOW_DRAW_MS = 12;
    // Car marker style; the selected driver is larger with a red outline.
    var CAR = {size: 15, line: "#ffffff", width: 2, text: 11};
    var SELECTED = {size: 24, line: "#e10600", width: 4, text: 14};

    var state = {
        anchor: null,     // last server clock: {time, at, playing, speed}
        rate: null,       // observed replay seconds per real second
        data: null,       // position window from the server
        selected: null,   // driver chosen in the dropdown
        shown: null,      // replay time currently drawn
        lastFrame: null,
        lastDrawn: null,
        frameCount: 0,
        everyOther: false
    };

    function sync(clock, paused, speed, data, driver) {
        var now = performance.now();
        if (driver !== state.selected) {
            state.selected = driver || null;
            state.lastDrawn = null;  // redraw the markers even when paused
        }
        var time = Number(clock) || 0;
        var playing = !paused;
        speed = Number(speed) || 1;
        var previous = state.anchor;
        if (playing && previous && previous.playing && previous.speed === speed && time > previous.time) {
            var observed = (time - previous.time) / ((now - previous.at) / 1000);
            if (observed > 0 && observed < speed * 3) {
                state.rate = 0.9 * state.rate + 0.1 * observed;
            }
        } else {
            state.rate = speed;
        }
        state.anchor = {time: time, at: now, playing: playing, speed: speed};
        state.data = data || null;
        return window.dash_clientside.no_update;
    }

    function serverEstimate(now) {
        var a = state.anchor;
        if (!a.playing) {
            return a.time;
        }
        var lead = ((now - a.at) / 1000) * state.rate;
        return a.time + Math.min(lead, MAX_LEAD_SECONDS * a.speed);
    }

    function displayTime(now) {
        var a = state.anchor;
        var target = serverEstimate(now);
        var dt = state.lastFrame === null ? 0 : Math.min((now - state.lastFrame) / 1000, 0.1);
        state.lastFrame = now;
        if (state.shown === null || !a.playing || Math.abs(target - state.shown) > SEEK_SECONDS * a.speed) {
            state.shown = target;
        } else {
            // Advance at playback rate and remove the remaining error smoothly;
            // the step is never negative, so cars do not move backwards.
            var step = dt * state.rate + (target - state.shown) * Math.min(1, dt / CORRECTION_SECONDS);
            state.shown += Math.max(0, step);
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
        var out = {x: [], y: [], text: [], color: [], size: [], line: [], width: [], font: []};
        var chosen = null;
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
            if (d.drivers[k] === state.selected) {
                chosen = {x: x, y: y, k: k};
                continue;
            }
            push(out, x, y, d.drivers[k], d.colors[k], CAR);
        }
        // Drawn last, so it stays on top of the other cars.
        if (chosen) {
            push(out, chosen.x, chosen.y, d.drivers[chosen.k], d.colors[chosen.k], SELECTED);
        }
        return out;
    }

    function push(out, x, y, name, color, style) {
        out.x.push(x);
        out.y.push(y);
        out.text.push(name);
        out.color.push(color);
        out.size.push(style.size);
        out.line.push(style.line);
        out.width.push(style.width);
        out.font.push(style.text);
    }

    function frame(now) {
        window.requestAnimationFrame(frame);
        if (!state.anchor || !state.data || !window.Plotly) {
            return;
        }
        var t = displayTime(now);
        state.frameCount += 1;
        if (state.everyOther && state.frameCount % 2) {
            return;
        }
        var graph = document.querySelector("#replay-track .js-plotly-plot");
        if (!graph || !graph.data || graph.data.length < 2) {
            return;
        }
        var carTrace = graph.data.length - 1;
        // A freshly drawn figure has an empty car trace; redraw even when paused.
        var empty = !graph.data[carTrace].x || graph.data[carTrace].x.length === 0;
        if (t === state.lastDrawn && !empty) {
            return;
        }
        var cars = positions(state.data, t);
        if (!cars) {
            return;
        }
        var started = performance.now();
        window.Plotly.restyle(
            graph,
            {
                x: [cars.x], y: [cars.y], text: [cars.text], "marker.color": [cars.color],
                "marker.size": [cars.size], "marker.line.color": [cars.line],
                "marker.line.width": [cars.width], "textfont.size": [cars.font]
            },
            [carTrace]
        );
        state.everyOther = performance.now() - started > SLOW_DRAW_MS;
        state.lastDrawn = t;
    }

    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        replay: {sync: sync}
    });
    window.requestAnimationFrame(frame);
})();
