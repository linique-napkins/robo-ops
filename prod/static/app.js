// SO101 Demo UI

let ws = null;
let currentState = "disconnected";

// --- API ---

async function api(method, path) {
    try {
        const res = await fetch("/api/" + path, { method });
        const data = await res.json();
        if (!data.ok && data.error) status("Error: " + data.error);
        if (data.state) updateState(data.state);
        return data;
    } catch (e) {
        status("Request failed: " + e.message);
        return { ok: false };
    }
}

async function doConnect() {
    status("Connecting to robot...");
    const data = await api("POST", "connect");
    if (data.ok) {
        startCameras();
        connectWS();
        status("Connected");
    }
}

async function doDisconnect() {
    status("Disconnecting...");
    await api("POST", "disconnect");
    stopCameras();
    disconnectWS();
    status("Disconnected");
}

async function doStow() {
    status("Stowing...");
    await api("POST", "stow");
}

async function doTogglePause() { await api("POST", "pause"); }

async function doStartReplay(loop) {
    const name = document.getElementById("sel-recording").value;
    if (!name) return status("Select a recording first");
    status("Starting replay: " + name);
    await api("POST", "replay/" + encodeURIComponent(name) + (loop ? "?loop=true" : ""));
}

async function doStopReplay() {
    const name = document.getElementById("sel-recording").value || "_";
    await api("POST", "replay/" + encodeURIComponent(name) + "/stop");
}

async function doStartInference() {
    const model = document.getElementById("sel-model").value;
    if (!model) return status("Select a model first");
    status("Loading model: " + model + "...");
    await api("POST", "infer/" + encodeURIComponent(model));
}

async function doStopInference() {
    const model = document.getElementById("sel-model").value || "_";
    await api("POST", "infer/" + encodeURIComponent(model) + "/stop");
}

async function doStartTeleop() { await api("POST", "teleop/start"); }
async function doStopTeleop() { await api("POST", "teleop/stop"); }

// --- State ---

function updateState(state) {
    currentState = state.state;

    // Badge
    const badge = document.getElementById("state-badge");
    let label = state.state;
    if (state.operation) label = state.state + " (" + state.operation + ")";
    if (state.paused) label += " [PAUSED]";
    badge.textContent = label;
    badge.className = "state-badge state-" + state.state;

    // Button states
    const connected = state.state !== "disconnected";
    const idle = state.state === "idle";
    const active = ["replaying", "inferring", "teleop"].indexOf(state.state) >= 0;

    document.getElementById("btn-connect").disabled = connected;
    document.getElementById("btn-disconnect").disabled = !connected;
    document.getElementById("btn-stow").disabled = !connected || state.state === "stowing";
    document.getElementById("btn-pause").disabled = !active;
    document.getElementById("btn-pause").textContent = state.paused ? "Resume" : "Pause";

    document.getElementById("btn-replay").disabled = !idle;
    document.getElementById("btn-replay-loop").disabled = !idle;
    document.getElementById("btn-replay-stop").disabled = state.state !== "replaying";

    document.getElementById("btn-infer").disabled = !idle;
    document.getElementById("btn-infer-stop").disabled = state.state !== "inferring";

    document.getElementById("btn-teleop").disabled = !idle;
    document.getElementById("btn-teleop-stop").disabled = state.state !== "teleop";

    // Populate dropdowns once
    if (state.available_recordings) populateSelect("sel-recording", state.available_recordings);
    if (state.available_models) populateModelSelect("sel-model", state.available_models);
}

function populateSelect(id, items) {
    const sel = document.getElementById(id);
    if (sel.options.length > 1) return;
    items.forEach(function(item) {
        const opt = document.createElement("option");
        opt.value = item;
        opt.textContent = item;
        sel.appendChild(opt);
    });
}

function populateModelSelect(id, models) {
    const sel = document.getElementById(id);
    if (sel.options.length > 1) return;
    Object.keys(models).forEach(function(key) {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = models[key];
        sel.appendChild(opt);
    });
}

// --- Cameras ---

function startCameras() {
    var ts = Date.now();
    document.getElementById("cam-left").src = "/stream/left?t=" + ts;
    document.getElementById("cam-top").src = "/stream/top?t=" + ts;
    document.getElementById("cam-right").src = "/stream/right?t=" + ts;
}

function stopCameras() {
    document.getElementById("cam-left").src = "";
    document.getElementById("cam-top").src = "";
    document.getElementById("cam-right").src = "";
}

// --- WebSocket ---

function connectWS() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws/state");
    ws.onmessage = function(e) { updateState(JSON.parse(e.data)); };
    ws.onclose = function() { ws = null; };
}

function disconnectWS() {
    if (ws) { ws.close(); ws = null; }
}

// --- Status ---

function status(msg) {
    document.getElementById("status-bar").textContent = msg;
}

// --- Init ---

(async function() {
    var data = await api("GET", "state");
    if (data && data.state && data.state !== "disconnected") {
        startCameras();
        connectWS();
    }
})();
