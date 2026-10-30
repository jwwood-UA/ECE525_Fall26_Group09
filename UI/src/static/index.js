// Escape HTML special characters and make invisible whitespace visible
const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const escape_token = (tok) =>
    String(tok)
        .replace(/[&<>"']/g, (ch) => HTML_ESCAPES[ch])
        .replace(/\n/g, "↵")   // newline -> ↵
        .replace(/\t/g, "→");  // tab -> →

// Color scale shared by the grid cells and the color bar.
// prob_scale maps probability (0-1) to color intensity (0-1); swap in Math.sqrt(p) to emphasize small values.
const COLOR_LOW = "white";
const COLOR_HIGH = "#1d4ed8";
const prob_scale = (p) => Math.min(Math.max(Number(p) || 0, 0), 1);

// Probability (0-1) -> background color on a white-to-blue scale
const prob_color = (p) => {
    const pct = Math.round(prob_scale(p) * 100);
    return `color-mix(in oklab, ${COLOR_HIGH} ${pct}%, ${COLOR_LOW})`;
};

// Vertical color bar: gradient plus tick labels placed with the same scale as the cells
const build_colorbar = (el, ticks = [0, 0.25, 0.5, 0.75, 1]) => {
    let html = `<div class="colorbar-body">`;
    html += `<div class="colorbar-gradient" style="background: linear-gradient(to top in oklab, ${COLOR_LOW}, ${COLOR_HIGH})"></div>`;
    html += `<div class="colorbar-ticks">`;
    for (const t of ticks) {
        html += `<span class="colorbar-tick" style="bottom:${prob_scale(t) * 100}%">${t}</span>`;
    }
    html += `</div></div>`;
    el.innerHTML = html;
};

let logit_state = null;

const fill_logit_lens = (logit_data) => {
    unpin_popup();                       // old cells are about to be replaced
    // Each cell is N predictions followed by the true-next-token entry,
    // so N comes from the data itself rather than what was requested
    const N = logit_data.top_k;
    logit_state = {"data":logit_data,"N":N};
    const nLayers = logit_data.n_layers;
    let logit_grid = "<thead><tr><th>Token</th>";
    for (let l=0; l<nLayers; l++) logit_grid += `<th>${l}</th>`;
    logit_grid += "</tr></thead><tbody>";
    let tok = -1;
    for(const token_data of logit_data.tokens){
        tok+=1;
        const token = escape_token(token_data.text);
        logit_grid += `<tr><th><span class="tok">${token}</span></th>`;
        let lay = -1;
        for(const layer_data of logit_data.results.slice(tok*nLayers,(tok+1)*nLayers)){
            lay+=1;
            const top_p = Number(layer_data.predictions[0].probability);
            const text_class = top_p > 0.55 ? " dark" : ""; // white text on dark backgrounds
            logit_grid += `<td data-pos="${tok}" data-layer="${lay}" style="background:${prob_color(top_p)}"><div class="cell${text_class}">`; // open layer box
            logit_grid += `<span class="top-pred tok">${escape_token(layer_data.predictions[0].token)}</span>`; // show highest confidence
            // Add banner to see true next token's position if there is a next token
            if(layer_data.reference_token){
                logit_grid += `<p class="true-next-token-banner"><span class="tok">${escape_token(layer_data.reference_token.token)}</span>: ${Number(layer_data.reference_token.rank)}</p>`;
            }
            logit_grid += "</div></td>"; // close layer box
        }

        logit_grid += "</tr>"; // close token row
    }
    logit_grid += "</tbody>" // close table

    // fetch grid element and insert new table data
    const grid = document.querySelector('#logit-content .grid');
    grid.innerHTML = logit_grid;
    size_cells(grid);
    build_colorbar(document.querySelector('#logit-colorbar'));

};

// Let cells take their natural size, find the largest, make every cell that square size
const size_cells = (table) => {
    table.classList.add("measuring");
    let max = 0;
    for (const c of table.querySelectorAll(".cell")) {
        max = Math.max(max, c.offsetWidth, c.offsetHeight);
    }
    table.classList.remove("measuring");
    table.style.setProperty("--cell", Math.ceil(max) + "px");
};

let logit_prompt = null;
let logit_N_selector = null;

// ---------- Top-N popup ----------

let logit_popup = null;   // set on DOMContentLoaded
let pinned_td = null;     // cell whose popup is pinned open, or null

// 0.62 -> "62.0%", 0.0004 -> "0.040%": keep small probabilities readable
const fmt_pct = (p) => {
    const pct = Number(p) * 100;
    return pct >= 10 ? `${pct.toFixed(1)}%` : `${pct.toPrecision(2)}%`;
};

// One row: rank | token | probability bar | percentage
const popup_row = (rank, tok, p, is_truth) => `
    <li class="popup-row${is_truth ? " hit" : ""}">
        <span class="popup-rank">${rank}</span>
        <span class="popup-tok">${escape_token(tok)}</span>
        <span class="popup-bar-track"><span class="popup-bar" style="width:${prob_scale(p) * 100}%"></span></span>
        <span class="popup-pct">${fmt_pct(p)}</span>
    </li>`;

const show_popup = (td) => {
    const { data, N } = logit_state;
    const pos = +td.dataset.pos;
    const layer = +td.dataset.layer;
    const celldat = data.results[pos*data.n_layers + layer];
    const truthdat = celldat.reference_token;
    const has_truth = truthdat!==null;

    const rows = [];
    for (let i = 0; i < N; i++) {
        const tok = celldat.predictions[i].token;
        const p = celldat.predictions[i].probability;
        rows.push(popup_row(i + 1, tok, p, has_truth && tok === truthdat.token));
    }
    // True next token ranked outside the top N: show it below a separator
    if (has_truth && truthdat.rank > N) {
        rows.push(`<li class="popup-sep"></li>`);
        rows.push(popup_row(`#${truthdat.rank}`, truthdat.token, truthdat.probability, true));
    }

    const note = has_truth
        ? `<span class="popup-key"></span>actual next token`
        : `No next token in the prompt`;
    const pin_hint = td === pinned_td ? " · click or Esc to close" : " · click to pin";

    logit_popup.innerHTML = `
        <div class="popup-title">
            Layer ${layer}
            <span class="popup-sub">after <span class="popup-tok">${escape_token(data.tokens[pos].text)}</span></span>
        </div>
        <ol class="popup-rows">${rows.join("")}</ol>
        <div class="popup-note">${note}${pin_hint}</div>`;

    place_popup(td);
};

const hide_popup = () => logit_popup && logit_popup.classList.add("inactive");

const pin_popup = (td) => {
    unpin_popup();
    pinned_td = td;
    td.classList.add("pinned");
    logit_popup.classList.add("pinned");
    show_popup(td);                                // redraw with the "close" hint
};

const unpin_popup = () => {
    if (pinned_td) pinned_td.classList.remove("pinned");
    pinned_td = null;
    if (logit_popup) logit_popup.classList.remove("pinned");
    hide_popup();
};

const place_popup = (td) => {
    const GAP = 8;
    const MARGIN = 8;
    logit_popup.classList.remove("inactive");      // must be visible to measure
    const rect = td.getBoundingClientRect();
    const w = logit_popup.offsetWidth;
    const h = logit_popup.offsetHeight;

    let left = rect.right + GAP;                   // prefer the right of the cell
    if (left + w > window.innerWidth - MARGIN) {
        left = rect.left - GAP - w;                // flip to the left
    }
    let top = rect.top;
    top = Math.min(top, window.innerHeight - MARGIN - h);
    top = Math.max(top, MARGIN);

    logit_popup.style.left = `${Math.max(left, MARGIN)}px`;
    logit_popup.style.top = `${top}px`;
};

// ---------- Running the model ----------

const N_DEFAULT = 5;
const N_MIN = 1;
const N_MAX = 10;

// Read the N input, clamp to [N_MIN, N_MAX], and write the clean value back into the box
const read_N = () => {
    let N = Math.round(logit_N_selector.valueAsNumber);
    if (!Number.isFinite(N)) N = N_DEFAULT;          // empty or non-numeric input
    N = Math.min(Math.max(N, N_MIN), N_MAX);
    logit_N_selector.value = N;
    return N;
};

let lens_controller = null;   // aborts a request when a newer one replaces it
let last_timing = null;       // { round_trip_ms } for the latency display

const run_lens = (prompt, N) => {
    const t0 = performance.now();
    if (!prompt.trim()) return;
    if (lens_controller) lens_controller.abort();     // drop the stale request
    lens_controller = new AbortController();
    const controller = lens_controller;

    const run_btn = document.getElementById("logit-run");
    run_btn.disabled = true;
    run_btn.textContent = "Running…";

    fetch("/api/logit_lens", {                         // must match the route in app.py
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, top_k: N }),
        signal: controller.signal,
    })
        .then(response => {
            if (!response.ok) throw new Error(`Status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            fill_logit_lens(data);
            const full_latency = performance.now()-t0;
            console.log(`Logit Lens Request:\nCompute Latency: ${data.latency}ms\nFull Latency: ${full_latency}ms\nNetwork and UI Latency: ${full_latency-data.latency}ms\n`);
        })
        .catch(err => {
            if (err.name === "AbortError") return;     // replaced by a newer request
            console.error("Logit lens request failed:", err);
        })
        .finally(() => {
            if (controller !== lens_controller) return; // a newer request owns the button
            run_btn.disabled = false;
            run_btn.textContent = "Run";
        });
};

document.addEventListener("DOMContentLoaded", () => {
    logit_prompt = document.getElementById("logit-prompt");
    logit_N_selector = document.getElementById("logit-count-input");
    logit_popup = document.getElementById("logit-popup");
    const logit_table = document.querySelector("#logit-content .grid");
    const logit_scroll = document.querySelector("#logit-content .grid-scroll");

    // Hover preview (ignored while a popup is pinned)
    logit_table.addEventListener("pointerover", (e) => {
        if (pinned_td || !logit_state) return;
        const td = e.target.closest("td[data-pos]");
        if (td) show_popup(td);
    });
    logit_table.addEventListener("pointerleave", () => {
        if (!pinned_td) hide_popup();
    });

    // Click a cell to pin/unpin; click anywhere else to unpin
    document.addEventListener("click", (e) => {
        if (e.target.closest("#logit-popup")) return;          // clicks inside the pinned popup
        const td = e.target.closest("#logit-content td[data-pos]");
        if (td && logit_state) {
            td === pinned_td ? unpin_popup() : pin_popup(td);
        } else if (pinned_td) {
            unpin_popup();
        }
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") unpin_popup();
    });

    // Keep a pinned popup attached to its cell; hide hover previews
    const follow = () => (pinned_td ? place_popup(pinned_td) : hide_popup());
    logit_scroll.addEventListener("scroll", follow);
    window.addEventListener("resize", follow);

    // Enter in either box or clicking Run submits the form
    const logit_form = document.getElementById("logit-form");
    logit_form.addEventListener("submit", (ev) => {
        ev.preventDefault();                           // stop the page from reloading
        run_lens(logit_prompt.value, read_N());
    });

    /*
    // Changing N re-runs the current prompt; debounced so spinner clicks don't flood the server
    let n_timer = null;
    logit_N_selector.addEventListener("change", () => {
        const N = read_N();
        clearTimeout(n_timer);
        n_timer = setTimeout(() => run_lens(logit_prompt.value, N), 300);
    });
    */
    /*
    // prefill with placeholder data
    fetch('./sample.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            fill_logit_lens(data);
        })
        .catch(error => console.error('Error loading table:', error));
    */
});
