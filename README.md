# CAP → Qwen Reinforcement Learning with Verifiable Rewards (RLVR)

Engineering guide for turning the **Caselaw Access Project (CAP)** corpus + **Qwen-3-14B** into a continuously-improving legal LLM with fully deterministic rewards using **GRPO (Group Relative Policy Optimization)**.

## Project Structure

```
cap-rlvr/
├── README.md                    # This file
├── docs/
│   └── cap_rlvr_grpo_plan.md   # Comprehensive implementation plan
├── scripts/
│   ├── vast_setup.sh           # Remote system setup script
│   ├── prep_utils.py           # Shared utilities for data preparation
│   ├── prep_holding_task.py    # Generate holding selection tasks
│   ├── prep_bluebook_task.py   # Generate citation format tasks
│   ├── prep_summarise_task.py  # Generate IRAC summarization tasks
│   ├── prep_retrieval_task.py  # Generate case retrieval tasks
│   └── prep_entail_task.py     # Generate case relationship tasks
├── downloads/
│   ├── cli_download.py         # Robust CLI-based dataset download
│   ├── robust_download.py      # Alternative download with retry logic
│   └── streaming_download.py   # Memory-efficient streaming download
└── .env                        # Environment configuration
```

## Quick Start

1. **Setup Environment**: Use `scripts/vast_setup.sh` on a remote system with sufficient storage (80GB+ for CAP dataset)
2. **Download Dataset**: Run `downloads/cli_download.py` for robust CAP dataset acquisition
3. **Prepare Tasks**: Execute all `scripts/prep_*.py` scripts to generate training data
4. **Train Model**: Follow the GRPO training pipeline in `docs/cap_rlvr_grpo_plan.md`

## Key Features

- **Robust Dataset Download**: Multiple approaches for handling 78GB CAP dataset with resume capability
- **Multi-Task Training Data**: 5 legal reasoning tasks (holdings, citations, summaries, retrieval, relationships)
- **Verifiable Rewards**: Deterministic scoring functions for each task type
- **Process Supervision**: GRPO training with group-based reward comparisons
- **Production Ready**: Quantization, serving, and deployment pipeline included

## Legal Tasks Generated

1. **Holding Selection**: Multiple-choice questions identifying correct legal holdings
2. **Bluebook Citations**: Fill-in-the-blank citation format completion
3. **IRAC Summaries**: Structured case summarization using Issue-Rule-Application-Conclusion
4. **Case Retrieval**: Finding analogous cases based on legal concepts
5. **Relationship Classification**: Determining how cases relate (overrule, distinguish, affirm, etc.)

## Dataset

Based on the **Caselaw Access Project (CAP)** containing millions of US court decisions, processed into structured training tasks for legal reasoning.

See `docs/cap_rlvr_grpo_plan.md` for the complete implementation plan and training details.