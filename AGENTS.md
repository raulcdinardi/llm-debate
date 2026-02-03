Please follow these instructions I think will benefit the long-term quality of code and keep me in the loop with the project. Sometimes when these are not followed, it’s frustrating, as many hours can be wasted. When they are followed, normally it’s very productive and research is expedited. The code produced by the session with you can be trusted, as I will have a mental model of what’s going on, which is important. I’m working on Reward Hacking, as it’s relevant to x-risks (sometimes I will be working on something different and there’s a chance I forget to change this file after I’m done with my Reward Hacking paper; in those cases also follow these instructions—they are generally very good).

AUTONOMY LEVELS: Different work warrants different caution.

1. Merge-destined / research-interpretation code: Rigorous. Follow all rules. Ask before non-obvious decisions.

2. Debugging / exploration: More autonomy. Try things, report back with results and methodology.

3. Throwaway / agentic testing: High autonomy. Just do it.

4. Agentic busy work (setup, running experiments, infrastructure): Maximum autonomy. But autonomy is scoped—if you hit issues requiring changes that would normally need approval (e.g., changing experiment parameters), escalate or flag; don’t inherit the busy-work autonomy level for those decisions.

State which mode you’re in. Flag mode transitions. If uncertain, ask.

New projects: Ask if it’s a side project. Side projects assume high autonomy by default, scaling back as the project grows and becomes more serious.

Always when you ask a qeustion you should include your expected answer and if I dont answer the question assuem the expected answer is correct.

COST AWARENESS: Cheap actions (local tests, USP_VM GPU boxes) can run autonomously. Expensive actions need more consideration. Costs include:

* Compute: <5090 ≈ free, 5090 very low, 6000 Pro low, H100+ not negligible
* Time opportunity: a 5-minute script is expensive in attention/blocking terms
* Bloat: context window and codebase clutter from low-value actions

Prefer high-EV actions, not just low-cost ones. If you’re doing many small debug runs each revealing one issue, consider switching to one diagnostic branch that surfaces everything at once. Notice when you’re on a bad path and switch.

If running many cheap actions in sequence, warn the user—volume has cost even if each action is cheap.

DECISIONS LOG: Maintain /decisions.md—append-only, written immediately as things happen.

Each session: generate a session ID at start, mark all entries with it.

For each entry:

* Timestamp + elapsed time for long operations
* Session ID
* Decision/action/problem

Extremely dense, not prosaic. If the file gets large, start a new one (decisions-2.md, etc.), don’t reorganize.

For detailed write-ups or throwaway tasks that would bloat the main log: write to a separate .md file, add a one-line pointer in decisions.md (“see /decisions/debug-memory-issue-0602.md”). Keeps the main log scannable.

CODE PRACTICES:
* Do not program defensively (e.g., try/except blocks, `.get()` with default values). I mostly write code that I want to fail visibly (asserts are good but prone to unnecessary spamming; if something benefits from an assert, use it).
* Keep me in the loop. Before trying out changes, delineate a plan and summarize it to me with key decision points and why you choose what you choose. (Autonomy level assumed relevant to level of detail; for high autonomy this should be reported after genuinely trying to complete the task.)
* Try to be precise instead of sloppy (bloat cost). Building good code and keeping the system complexity manageable is **way** better long-term than myopically finishing your task. This, of course, depends on the disposableness of the task. For example, you don’t need to worry too much for disposable scripts or quick tasks that aren’t part of a bigger project. You can ask me about the scope of the task if it’s pretty ambiguous.
* Related to the previous point, try to follow the Unix philosophy, where each script/file does one thing and does it well. E.g., a training script should do training and log it, then a visualizer script should read the logs and visualize them, instead of “online” live visualization.
* Never do something in a way I don’t expect. E.g., I ask you to do an experiment with design X but that’s infeasible or hard, so you do a very related experiment. Even if you tell me, I might not read that part of the output you did. Always be **very** explicit about results, and if possible explain where you did something you believe I didn’t expect. Explain your plan as well as alternatives. (The frequency of reporting depends on autonomy level, so in a very autonomous task you should leave all the reporting to the end.)
* If the requessenset implies a larger refactor or design change, stop and ask before proceeding—no silent scope creep. (Big also depends on the importance of code; big refactors in throwaway code are smaller than refactoring a very core small function at the heart of research code.)
* State assumptions explicitly (inputs, shapes, file paths, expected invariants, performance constraints).
* Report changes as “what changed + why” (not just “done”).
* Prefer minimal diffs: 1) implement the minimal change satisfying the request 2) verify it works 3) refactor or generalize only after validation. Assume there is significant “wiring” cost; think of facilitated variation where weak regulatory linkage and modularity helps with evolvability.
* Aim for reproducibility (when not in conflict with standard practice, common sense, the goal of the project, etc.): set and log RNG seeds; log full config; write outputs with deterministic names.
* Run simple validations (except when it will cost money, e.g. it hits an API; then suggest running a minimal validation. If there’s an idle SPU instance you can start a minimal validation without suggesting.)
* Comments are for future readers (including LLM agents) who read code fluently but lack conversation context. Each comment should add information not recoverable from the code—why this approach, what shaped the decision, what non-obvious things it depends on. Prolix language wastes context window tokens; dense comments don’t. Roughly 30% comment-to-code ratio is fine if they’re high-value—the cost is verbosity, not volume. You can add comments to legacy code but you need to be certain of the information correctness in the comment.
* When reporting unexpected behavior, separate OBSERVED (what literally happened—logs, outputs, traces) from HYPOTHESIZED (your interpretation). Include what would confirm/refute the hypothesis. Don’t collapse these.
* Before fixing a bug, establish the causal chain. What’s the symptom? What are hypotheses? What evidence distinguishes them? Don’t propose a fix until you’ve gone at least 2 “whys” deep. If the root cause is unclear, say so rather than fixing the proximate symptom.
* For exploratory work (intrusive logging, instrumentation, throwaway tests), work in a disposable branch. The output is understanding, not code. Don’t let exploratory hacks leak into persistent code.
sense* When making design decisions, state the tradeoff: “I’m trading X for Y, assuming Z holds. A more thorough approach would be [alternative] at cost [C].” Let me decide if the tradeoff is acceptable.
* If you’re uncertain about project context, architectural intent, or why something exists—ask rather than guess. “I don’t know why this is structured this way” is better than a confident wrong assumption. Also, some assumptions are made by other LLM agents that coded the code (or even yourself); if some assumption of the code disagrees with the purpose of the project or what you infer my mental model is, flag that.
* If deviating from the literal request for any reason, put DEVIATION: [what and why] at the top of your response, not buried in explanation.
* When starting long-running processes (locally or on remote machines): test statically/cheaply before running, and actually verify they’re working—don’t just confirm they started. Check for expected outputs/logs/behavior. Be aware you don’t have good time intuition—if you check too early and see nothing, that might mean warmup, not failure; consider checking again later and looking at the real-world time or how long the process has been running for.

Very practical advice:

* LMs are very reliant on the token template they use during training; i.e., the chat template should exactly match. Instruction-tuned models should be given instructions, not completions like base models are given.
* Read tinker-cookbook/llms-full.txt if working with the Tinker API or the local API.
* The local and cloud APIs should match; flag STRONGLY discrepancies.

Thanks :)

Excerpts of the Wikipedia page of Facilitated Variation:

The theory of facilitated variation consists of several elements.[1][2] Organisms are built from a set of highly conserved modules called “core processes” that function in development and physiology, and have remained largely unchanged for millions (in some instances billions) of years. Genetic mutation leads to regulatory changes in the package of core components (i.e. new combinations, amounts, and functional states of those components) exhibited by an organism. Finally, the altered combinations, amounts, and states of the conserved components function to develop and operate a new trait on which natural selection acts. Because of their modular organization, adaptability (e.g. arising through exploratory processes) and compartmentation, developmental systems tend to produce facilitated (i.e. functional and adaptive) phenotypic variation when challenged by genetic mutation or novel environmental conditions.

Animals are built from a toolkit of components (e.g. like LEGO bricks).

Weak regulatory linkage
Different core processes become linked, through differential regulation, in different combinations, and operate in different amounts, states, times, and places, to generate new anatomical and physiological traits. These regulatory linkages can be made and changed easily, a phenomenon that Kirschner and Gerhart call “weak regulatory linkage”. Regulatory signals can switch on and off the core components to elicit complex responses. Although the signal seems to control the response, typically the responding core process can produce the output by itself but inhibits itself from doing so. All the signal does is interfere with this self-inhibition. Regulatory change is easily effected because conserved core processes have switch-like behavior and alternative outputs already built into them, which means that regulation does not need to coevolve with the functional output.

Example: evolution of the wing
Gerhart and Kirschner[2] give the example of the evolution of a bird or bat wing from a tetrapod forelimb. They explain how, if bones undergo regulatory change in length and thickness as a result of genetic mutation, the muscles, nerves and vasculature will accommodate to those changes without themselves requiring independent regulatory change. Studies of limb development show that muscle, nerve, and vascular founder cells originate in the embryonic trunk and migrate into the developing limb bud, which initially contains only bone and dermis precursors. Muscle precursors are adaptable; they receive signals from developing dermis and bone and take positions relative to them, wherever they are. Then, as noted previously, axons in large numbers extend into the bud from the nerve cord; some fortuitously contact muscle targets and are stabilized, and the rest shrink back. Finally, vascular progenitors enter. Wherever limb cells are hypoxic, they secrete signals that trigger nearby blood vessels to grow into their vicinity. Because of the adaptability conferred by exploratory processes, the co-evolution of bones, muscles, nerves and blood vessels is not required. Selection does not have to coordinate multiple independently varying parts. This not only means that viable phenotypes can easily be generated with little genetic change, but also that genetic mutations are less likely to be lethal, that large phenotypic changes can be favored by selection, and that phenotypic variation is functional and adaptive (i.e. “facilitated”).
