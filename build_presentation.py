"""
Build a self-contained HTML slide deck (PRESENTATION.html) for the professor.

Embeds every figure as base64 so the single file works offline / by email.
Navigation: arrow keys / space / click; press 'f' for fullscreen, 'o' for overview.
Regenerate any time with:  ../.venv/bin/python build_presentation.py
"""
import base64, os

HERE = os.path.dirname(os.path.abspath(__file__))


def img(rel):
    with open(os.path.join(HERE, rel), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


# (title, html-body) per slide. Figures embedded inline.
SLIDES = []


def slide(title, body, cls=""):
    SLIDES.append((title, body, cls))


def figure(rel, caption=""):
    cap = f'<div class="cap">{caption}</div>' if caption else ""
    return f'<img src="{img(rel)}" alt="{rel}"/>{cap}'


# ---------------------------------------------------------------- 1. TITLE
slide("", """
<div class="title-block">
  <h1>Cyclic dominance on complex networks</h1>
  <h2>A Potts&nbsp;(q=3) + Rock&ndash;Paper&ndash;Scissors model:<br/>recreation and new results</h2>
  <p class="sub">Monte&nbsp;Carlo ground truth &middot; homogeneous &amp; degree-based mean field &middot;
     network perturbations</p>
  <p class="meta">Summer reading project &mdash; progress review &middot; 2026-07-01</p>
</div>
""", "title")

# ---------------------------------------------------------------- 2. THE QUESTION
slide("The question", """
<ul class="big">
  <li>Agents on a network each play <b>Rock, Paper or Scissors</b> (a 3-state Potts spin).</li>
  <li>Interactions are <b>cyclically dominant</b>: Paper&nbsp;&gt;&nbsp;Rock&nbsp;&gt;&nbsp;Scissors&nbsp;&gt;&nbsp;Paper.
      Strength of the cycle = <b>&epsilon;</b>.</li>
  <li>Two possible collective fates:</li>
</ul>
<div class="two-col">
  <div class="card"><b>Order</b><br/>the network locks onto one strategy &mdash; consensus.
     Order parameter <b>m<sub>&psi;</sub> &rarr; 1</b>.</div>
  <div class="card"><b>Cycling</b><br/>strategies chase each other forever &mdash; no winner.
     <b>m<sub>&psi;</sub> &rarr; 0</b>.</div>
</div>
<p class="note"><b>Central question:</b> when does cyclic competition (&epsilon;) destroy order,
   and how does the <i>network structure</i> shift that boundary?</p>
""")

# ---------------------------------------------------------------- 3. MODEL / METHODS
slide("Model &amp; three levels of description", """
<div class="two-col">
  <div>
    <p><b>Payoff matrix</b> (cyclic-dominance strength &epsilon;):</p>
    <pre>P = I + &epsilon;&middot;skew
      R    P    S
  R [ 1  -&epsilon;   +&epsilon; ]
  P [ +&epsilon;   1  -&epsilon; ]
  S [ -&epsilon;  +&epsilon;   1 ]</pre>
    <p><b>Update:</b> Glauber/Metropolis at temperature T=0.65,
       accept with logistic(&Delta;U/T).</p>
    <p><b>Order parameter:</b> &psi; = r + p&middot;e<sup>i2&pi;/3</sup> + s&middot;e<sup>i4&pi;/3</sup>,
       &nbsp; m<sub>&psi;</sub> = |&langle;&psi;&rangle;<sub>t</sub>|.</p>
  </div>
  <div>
    <p><b>Three levels &mdash; cross-validated against each other:</b></p>
    <ul>
      <li><b>MC</b> &mdash; agent-level simulation on the actual graph. Ground truth.</li>
      <li><b>HMF</b> &mdash; homogeneous mean field: 3 ODEs, every node sees mean degree.</li>
      <li><b>DMF</b> &mdash; degree-based mean field: ODEs per degree class, uses full P(k).</li>
    </ul>
    <p><b>Graphs:</b> Erd&#337;s&ndash;R&eacute;nyi (ER, homogeneous) and Barab&aacute;si&ndash;Albert (BA, hubs).</p>
    <p><b>Engine:</b> C++20 (&minus;O3 &minus;march=native), xoshiro RNG, fanned across cores.</p>
  </div>
</div>
""")

# ---------------------------------------------------------------- 4. VALIDATION
slide("Step 0 &mdash; the engine is trustworthy", """
<p>Before any physics: the fast C++ engine is checked against an independent
   pure-Python Monte&nbsp;Carlo on the same graph (different RNG streams).</p>
<table class="data">
  <tr><th>&epsilon;</th><th>Python m<sub>&psi;</sub></th><th>C++ m<sub>&psi;</sub></th><th>verdict</th></tr>
  <tr><td>0.2</td><td>0.999</td><td>0.999</td><td class="ok">agree &mdash; ordered</td></tr>
  <tr><td>0.5</td><td>0.991</td><td>0.992</td><td class="ok">agree &mdash; ordered</td></tr>
  <tr><td>0.7</td><td>0.004</td><td>0.007</td><td class="ok">agree &mdash; cycling</td></tr>
  <tr><td>0.9</td><td>0.004</td><td>0.003</td><td class="ok">agree &mdash; cycling</td></tr>
</table>
<p class="note">Same regime at every &epsilon;, and <b>~20&times; faster</b> than Python.
   Every result below rests on this validated engine.</p>
""")

# ---------------------------------------------------------------- 5. RESULT 1 HMF sweep
slide("Result 1 &mdash; connectivity stabilises order", f"""
<div class="fig-row">
  <div class="fig">{figure("mean_field/hmf_sweep.png")}</div>
  <div class="txt">
    <p>Homogeneous mean field, sweeping &epsilon; at several mean degrees &langle;k&rangle;.</p>
    <p>The transition &epsilon;<sub>c</sub> (order&rarr;cycling) moves to <b>higher &epsilon;</b> as the
       network gets denser:</p>
    <table class="data small">
      <tr><th>&langle;k&rangle;</th><th>2</th><th>5</th><th>10</th><th>50</th><th>200</th></tr>
      <tr><td>&epsilon;<sub>c</sub></td><td>0.08</td><td>0.50</td><td>0.64</td><td>0.66</td><td>0.70</td></tr>
    </table>
    <p class="note">More neighbours &rarr; more mutual reinforcement &rarr; order survives
       stronger cyclic pressure.</p>
  </div>
</div>
""")

# ---------------------------------------------------------------- 6. RESULT 2 MC vs MF
slide("Result 2 &mdash; mean field vs the real thing", f"""
<div class="fig-row">
  <div class="fig two">{figure("mean_field/comparison_suite_ER_k10.png")}{figure("mean_field/comparison_suite_BA_k10.png")}</div>
  <div class="txt">
    <p>MC (ground truth) vs HMF vs DMF, at &langle;k&rangle;=10.</p>
    <ul>
      <li>Both mean fields <b>overestimate</b> the ordered phase: MC turns over at
          &epsilon;&asymp;0.52, mean field at &asymp;0.64.</li>
      <li><b>DMF beats HMF</b> on both graphs (lower RMSE vs MC):</li>
    </ul>
    <table class="data small">
      <tr><th>graph</th><th>RMSE HMF</th><th>RMSE DMF</th><th>DMF gain</th></tr>
      <tr><td>ER</td><td>0.338</td><td>0.334</td><td>0.004</td></tr>
      <tr><td>BA</td><td>0.336</td><td>0.328</td><td><b>0.008</b></td></tr>
    </table>
    <p class="note">DMF's advantage is <b>~2&times; larger on BA</b> &mdash; resolving the degree
       distribution pays off exactly where hubs make it heterogeneous.</p>
  </div>
</div>
""")

# ---------------------------------------------------------------- 7. RESULT 3 phase diagram
slide("Result 3 &mdash; the phase diagram", f"""
<div class="fig-row">
  <div class="fig two">{figure("phase_diagram/phase_diagram_ER.png")}{figure("phase_diagram/phase_diagram_BA.png")}</div>
  <div class="txt">
    <p>Full (&langle;k&rangle; &times; &epsilon;) sweep &mdash; <b>520 simulations per graph</b>,
       fanned across 16 cores in ~6&nbsp;s.</p>
    <p>The order&ndash;cycling boundary bends upward: denser networks stay ordered
       to higher &epsilon; (&epsilon;<sub>c</sub>: 0 &rarr; 0.64 &rarr; 0.72 as &langle;k&rangle; grows).</p>
    <p class="note">ER and BA give <b>nearly identical</b> boundaries &mdash; for MC dynamics the
       <i>average</i> degree matters more than the shape of P(k).</p>
  </div>
</div>
""")

# ---------------------------------------------------------------- 8. RESULT 4 dynamics/FSS
slide("Result 4 &mdash; dynamics &amp; finite-size scaling", f"""
<div class="fig-row">
  <div class="fig two">{figure("dynamics/fss.png")}{figure("dynamics/ternary.png")}</div>
  <div class="txt">
    <p><b>Finite-size scaling</b> (left): the transition <b>sharpens</b> and its midpoint
       <b>converges</b> as the system grows &mdash; &epsilon;<sub>c</sub>: 0.57&rarr;0.53&rarr;0.51&rarr;0.51
       for N = 200&rarr;500&rarr;1000&rarr;2000.</p>
    <p><b>Phase-space portrait</b> (right): below &epsilon;<sub>c</sub> trajectories fall into a
       <b>corner</b> (consensus); above it they settle onto a <b>limit cycle</b>
       (endless RPS chase).</p>
    <p class="note">Confirms a genuine phase transition, not a finite-size artefact.</p>
  </div>
</div>
""")

# ---------------------------------------------------------------- 9. DIVIDER: new work
slide("", """
<div class="title-block">
  <h1 class="accent">Beyond the recreation</h1>
  <h2>Three new perturbation studies &mdash; how robust is the ordered phase?</h2>
  <p class="sub">stubborn agents (zealots) &middot; hub targeting &middot; network damage (defects)</p>
</div>
""", "title divider")

# ---------------------------------------------------------------- 10. NEW: zealots
slide("New 1 &mdash; can a stubborn minority take over?", f"""
<div class="fig-row">
  <div class="fig">{figure("zealots/zealots.png")}</div>
  <div class="txt">
    <p>Add a fraction z of <b>zealots</b> permanently locked to <b>Rock</b>.</p>
    <ul>
      <li><b>Ordering phase:</b> a few Rock-zealots <b>back-fire</b> &mdash; by z&asymp;5% the free
          network flips not to Rock but to <b>Paper</b> (the strategy that <i>beats</i> Rock).
          Rock-conversion &rarr; 0.</li>
      <li><b>Cycling phase:</b> zealots induce only weak order (m<sub>&psi;</sub>&asymp;0.18 at z=0.2)
          and cannot pin their own strategy.</li>
    </ul>
    <p class="note">Counter-intuitive: in a cyclic system, pushing your strategy
       <b>feeds its predator</b>. Naive minority-takeover fails.</p>
  </div>
</div>
""")

# ---------------------------------------------------------------- 11. NEW: hubs
slide("New 2 &mdash; target the hubs", f"""
<div class="fig-row">
  <div class="fig">{figure("zealots/zealots_hubs.png")}</div>
  <div class="txt">
    <p>Same zealots, but placed on the <b>highest-degree nodes</b> of a BA network
       instead of at random.</p>
    <p>In the cycling phase, 10% <b>hub</b> zealots drive m<sub>&psi;</sub> to <b>0.72</b>,
       vs <b>0.08</b> for random placement &mdash; a <b>~9&times; amplification</b>.</p>
    <p class="note">Hubs control <b>whether</b> the network orders &mdash; but not <b>what</b> it
       orders on (still Paper, the predator). Structural leverage, not strategic control.</p>
  </div>
</div>
""")

# ---------------------------------------------------------------- 12. NEW: mixed + defects
slide("New 3 &mdash; competing factions &amp; network damage", f"""
<div class="fig-row">
  <div class="fig two">{figure("zealots/zealots_mixed.png")}{figure("defects/defects.png")}</div>
  <div class="txt">
    <p><b>Competing zealots</b> (left): equal Rock + Paper factions. At high z the free
       population goes to <b>Paper (0.90)</b>, not Scissors &mdash; Paper is reinforced by its own
       zealots <i>and</i> by the Rock-zealots it preys on.</p>
    <p><b>Network defects</b> (right): remove a fraction f of edges or nodes. The transition
       slides to lower &epsilon; (&epsilon;<sub>c</sub>: 0.64&rarr;0.24 as f: 0&rarr;0.8), and
       <b>edge and node damage coincide</b> once matched by the resulting &langle;k&rangle;.</p>
    <p class="note">Order-stability depends on <b>effective &langle;k&rangle; only</b> &mdash; a clean,
       unifying result across two different kinds of disorder.</p>
  </div>
</div>
""")

# ---------------------------------------------------------------- 13. RIGOR
slide("Rigor &amp; reproducibility", """
<div class="two-col">
  <div>
    <p><b>One command reproduces everything:</b></p>
    <pre>./run_all.sh</pre>
    <p>rebuilds the engine &rarr; runs the validation test &rarr; runs all 16 simulation
       steps &rarr; regenerates every figure and data table.</p>
    <p><b>Everything is logged:</b> per-step logs in <code>logs/</code>, a machine-readable
       <code>logs/manifest.csv</code>, and a written <code>RUN_REPORT.md</code>.</p>
  </div>
  <div>
    <ul class="big">
      <li><b>16 / 16</b> steps pass, exit 0.</li>
      <li>Every figure &amp; CSV regenerated <b>byte-for-byte identical</b> to the committed
          version &mdash; fully deterministic seeds.</li>
      <li>Each figure ships with the <b>CSV behind it</b> (23 artifacts total).</li>
      <li>Version-controlled, pushed to a private repo.</li>
    </ul>
  </div>
</div>
""")

# ---------------------------------------------------------------- 14. STATUS + ASK
slide("Where we are &amp; what's next", """
<div class="two-col">
  <div>
    <p class="done"><b>Done</b></p>
    <ul>
      <li>Full recreation of the reference project (MC, HMF, DMF, phase diagram, FSS, dynamics).</li>
      <li>Three new perturbation studies with clean, non-obvious results.</li>
      <li>Reproducible, logged, documented, version-controlled.</li>
    </ul>
  </div>
  <div>
    <p class="next"><b>Next &mdash; and where HPC helps</b></p>
    <ul>
      <li>Larger N &amp; more seeds &rarr; <b>proper error bars</b> and a precise &epsilon;<sub>c</sub>(&langle;k&rangle;)
          critical curve.</li>
      <li>Finite-size scaling collapse to extract critical exponents.</li>
      <li>BA-vs-ER defect robustness &amp; time-varying disorder at scale.</li>
    </ul>
  </div>
</div>
<p class="note center">Current sweeps run in ~2&nbsp;min on a laptop. Production-quality
   statistics (many seeds &times; large N) is the natural case for <b>cluster access</b>.</p>
""")


# ------------------------------------------------------------------ RENDER
def render():
    sections = []
    for i, (title, body, cls) in enumerate(SLIDES):
        head = f'<h3 class="slide-title">{title}</h3>' if title else ""
        num = f'<div class="pagenum">{i+1} / {len(SLIDES)}</div>'
        sections.append(f'<section class="slide {cls}">{head}{body}{num}</section>')
    slides_html = "\n".join(sections)
    return TEMPLATE.replace("{{SLIDES}}", slides_html).replace("{{N}}", str(len(SLIDES)))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Cyclic dominance on networks &mdash; progress</title>
<style>
:root{ --bg:#0e1116; --fg:#e7ecf3; --dim:#9aa7b8; --accent:#5db0ff; --accent2:#ffcf6b;
       --card:#1a2230; --ok:#57d38c; }
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
  font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
#deck{height:100vh;overflow:hidden;position:relative}
.slide{position:absolute;inset:0;padding:4vh 5vw;display:none;
  flex-direction:column;justify-content:flex-start;overflow:auto}
.slide.active{display:flex}
.slide-title{font-size:2.1vw;margin:0 0 2.2vh;color:var(--accent);
  border-bottom:2px solid #26364d;padding-bottom:1vh;font-weight:700}
h1{font-size:4vw;margin:.2em 0;line-height:1.08}
h2{font-size:1.9vw;font-weight:500;color:var(--dim);margin:.4em 0}
p,li{font-size:1.35vw;line-height:1.5}
ul.big li{font-size:1.55vw;margin:.6em 0}
.sub{color:var(--accent2);font-size:1.5vw}
.meta{color:var(--dim);font-size:1.1vw;margin-top:3vh}
.title-block{margin:auto;text-align:center;max-width:80%}
.title.divider h1{color:var(--accent2)}
.accent{color:var(--accent2)}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:3vw;align-items:start}
.card{background:var(--card);border:1px solid #2a3546;border-radius:12px;
  padding:1.4vw 1.6vw;font-size:1.3vw}
.card b{color:var(--accent);font-size:1.5vw}
.fig-row{display:grid;grid-template-columns:1.15fr .85fr;gap:2.5vw;align-items:center;flex:1}
.fig{display:flex;gap:1vw;justify-content:center;align-items:center}
.fig img{max-width:100%;max-height:74vh;border-radius:10px;background:#fff;padding:6px}
.fig.two img{max-height:62vh}
.txt p:first-child{margin-top:0}
pre{background:#0a0d12;border:1px solid #26324a;border-radius:8px;padding:1em;
  font-size:1.15vw;color:#cfe3ff;overflow:auto}
code{background:#0a0d12;padding:.1em .4em;border-radius:5px;color:#cfe3ff}
table.data{border-collapse:collapse;margin:1.2vh 0;font-size:1.25vw}
table.data.small{font-size:1.1vw}
table.data th,table.data td{border:1px solid #2c3a52;padding:.35em .8em;text-align:center}
table.data th{background:#18202e;color:var(--accent)}
.ok{color:var(--ok);font-weight:600}
.note{background:#141c2a;border-left:4px solid var(--accent);padding:.8em 1.1em;
  border-radius:0 8px 8px 0;color:#d6e2f2;font-size:1.25vw;margin-top:1.6vh}
.note.center{text-align:center;border-left:none;border-top:4px solid var(--accent2)}
.cap{font-size:1vw;color:var(--dim);text-align:center;margin-top:.4vh}
.done{color:var(--ok);font-size:1.6vw;margin:0}
.next{color:var(--accent2);font-size:1.6vw;margin:0}
.pagenum{position:absolute;bottom:2vh;right:2.5vw;color:var(--dim);font-size:1vw}
#bar{position:fixed;top:0;left:0;height:4px;background:var(--accent);width:0;
  transition:width .2s;z-index:10}
#hint{position:fixed;bottom:1vh;left:2vw;color:#54607a;font-size:.9vw;z-index:10}
</style></head>
<body>
<div id="bar"></div>
<div id="deck">
{{SLIDES}}
</div>
<div id="hint">&larr;/&rarr; or space to navigate &middot; f = fullscreen &middot; o = overview</div>
<script>
const slides=[...document.querySelectorAll('.slide')];
let i=0, overview=false;
function show(n){ i=Math.max(0,Math.min(slides.length-1,n));
  slides.forEach((s,k)=>s.classList.toggle('active',k===i));
  document.getElementById('bar').style.width=((i+1)/slides.length*100)+'%';
  location.hash=i+1; }
function toggleOverview(){ overview=!overview;
  const d=document.getElementById('deck');
  if(overview){ d.style.overflow='auto'; d.style.height='auto';
    slides.forEach(s=>{s.style.position='relative';s.style.display='flex';
      s.style.height='100vh';s.style.borderBottom='3px solid #223';}); }
  else{ d.style.overflow='hidden'; d.style.height='100vh';
    slides.forEach(s=>{s.style.position='absolute';s.style.height='';s.style.display='';
      s.style.borderBottom='';}); show(i); }
}
document.addEventListener('keydown',e=>{
  if(overview && e.key!=='o' && e.key!=='f') return;
  if(['ArrowRight',' ','PageDown','ArrowDown'].includes(e.key)){show(i+1);e.preventDefault();}
  else if(['ArrowLeft','PageUp','ArrowUp'].includes(e.key)){show(i-1);e.preventDefault();}
  else if(e.key==='Home'){show(0);} else if(e.key==='End'){show(slides.length-1);}
  else if(e.key==='f'){ if(!document.fullscreenElement)document.documentElement.requestFullscreen();
      else document.exitFullscreen(); }
  else if(e.key==='o'){ toggleOverview(); }
});
document.getElementById('deck').addEventListener('click',e=>{ if(!overview) show(i+1); });
show((parseInt(location.hash.slice(1))||1)-1);
</script>
</body></html>"""


if __name__ == "__main__":
    html = render()
    out = os.path.join(HERE, "PRESENTATION.html")
    with open(out, "w") as f:
        f.write(html)
    kb = os.path.getsize(out) // 1024
    print(f"Wrote {out}  ({len(SLIDES)} slides, {kb} KB, self-contained)")
