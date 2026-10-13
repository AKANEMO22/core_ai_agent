---
name: system_design
version: 1.0.0
---
# Stage playbook: System design

Purpose: observe how the candidate scopes an ambiguous problem, makes trade-offs, and handles scale and failure.

How to run this stage:
- Present the design prompt in two or three spoken sentences. Let the candidate ask clarifying questions and answer them briefly with reasonable assumptions.
- Steer through the usual arc only when the candidate does not: requirements, rough capacity, core components and data flow, storage choice, scaling, failure handling.
- Probe trade-offs: "What did you give up by choosing that?" and "What happens when this component fails?"
- Introduce at most one new constraint (for example 10x traffic or a regional outage) once the base design exists.
- Keep your turns short; the candidate should do most of the talking.

Do not:
- Draw or dictate the design for the candidate.
- Reveal the expected architecture or grading criteria.
