**Overview**

I used AI tools (mainly ChatGPT and GitHub Copilot) as learning aids while developing this project. The goal was to clarify concepts, resolve bugs, and improve code quality. At every step, I made sure I understood the logic behind the code, rewrote parts as needed, and verified correctness through testing.

**How AI Assisted the Development Process
Debugging and troubleshooting**

A large part of the interaction with AI focused on diagnosing unexpected behavior: index misalignment, shifting logic for returns, clustering label issues, and pipeline errors. These tools helped me understand why certain problems occurred so I could fix them directly.

Understanding libraries and structuring the workflow

I frequently asked for clarifications on how to structure a clean sklearn pipeline, how to organise the project into modular files, and how to implement PCA, clustering, or walk-forward validation correctly. These explanations guided the architecture, but the design decisions and parameter choices were mine.

**Code refinement and consistency**

Most of the code in src/ began as a working version that I wrote myself, through the jupyter notebooks. AI tools helped me refactor it into a more consistent and readable form: improving naming, reducing repetition, reorganising functions, and making the overall structure cleaner. This refinement explains why the files share a coherent style.
However, all the logic, feature engineering choices, model behaviour, split definitions, strategy design, was implemented, tested, and adjusted by me.

**Template suggestions for standard patterns**

For common structures (e.g., evaluation functions, plotting boilerplate, or wrapper functions), I sometimes asked for a small template to avoid reinventing patterns that are widely used in data science. These templates were always adapted to the needs of the project and integrated only after I confirmed they behaved as expected.

**Documentation and readability**

AI helped draft or polish descriptive text, such as the project description and in-code comments, to make the project easier to understand.

**What AI Did Not Do**

It did not design the methodology or decide how regimes, PCA, clustering, or strategies should work.

It did not build the trading logic or define the walk-forward scheme.

It did not write code that I did not understand; every part of the final pipeline has been verified manually.

It did not produce a fully formed project that I simply copied. All components were rewritten, modified, extended, or debugged by me.

**Final Remarks**

The project reflects my own understanding of quantitative modelling, machine learning, and strategy evaluation. AI tools were used the same way one would use documentation, StackOverflow, or a tutor: to clarify concepts, accelerate debugging, and clean up code, while keeping full control over the design and implementation.