# SENTINEL-X — Defence Project Partner Attractiveness Analysis

## 1. Project

**SENTINEL-X** is a fully air-gapped, heterogeneous Edge AI architecture for reconnaissance and human-in-the-loop situational awareness in communication-degraded and EW-contested environments.

The core architecture is:

```text
AIRBORNE NODE
Camera
   ↓
Jetson Orin NX
   ├── Real-time CV / object localization
   ↓
MCXN947
   ├── independent lightweight CNN verification
   ↓
Jetson Orin NX
   ├── local multimodal RAG + VLM
   ├── contextual interpretation
   └── compact structured event
   ↓
Low-bandwidth tactical event message
   ↓
GROUND NODE
Raspberry Pi 5
   ├── local event/state processing
   ├── Whisper.cpp STT
   ├── local SLM
   └── Piper TTS
   ↓
Human operator
```

The project is therefore not simply a YOLO demo, a RAG demo, or a local LLM demo. Its central engineering proposition is the **integration of heterogeneous Edge AI processors into a distributed, disconnected, locally intelligent reconnaissance architecture**.

---

# 2. Why this can attract defence project partners

The strongest reason for interest is that SENTINEL-X addresses several requirements that are important in defence Edge AI at the same time:

- operation without cloud connectivity;
- operation when communications are degraded;
- low-bandwidth information exchange;
- processing close to the sensor;
- heterogeneous compute;
- independent verification of AI outputs;
- protection of sensitive models and knowledge;
- multimodal AI at the tactical edge;
- human-in-the-loop interaction;
- low-SWaP embedded implementation.

A defence company or research organisation does not necessarily need the prototype to be a finished military product.

What can attract them is a credible demonstration that a difficult architecture can actually be implemented on constrained hardware.

The project can therefore be presented as a **research and technology demonstrator**, rather than as a finished operational military system.

---

# 3. What makes the architecture interesting

## 3.1 Heterogeneous Edge AI

The system deliberately gives different processors different responsibilities.

### Jetson Orin NX

Responsible for:

- high-throughput computer vision;
- object localization;
- tracking;
- multimodal reasoning;
- local RAG;
- VLM inference.

### MCXN947

Responsible for:

- independent secondary classification;
- low-power inference;
- hardware-isolated execution;
- independent verification of candidate detections.

### Raspberry Pi 5

Responsible for:

- receiving compact event information;
- maintaining local state;
- speech recognition;
- local language-model interaction;
- speech synthesis.

This demonstrates that the designer understands **system-level Edge AI**, rather than simply running an AI model on one development board.

That distinction is important for defence-oriented engineering.

---

# 4. The most attractive architectural idea

The strongest part of the concept is the transformation:

```text
Continuous sensor data
        ↓
Local AI processing
        ↓
Structured intelligence
        ↓
Compact event
        ↓
Human operator
```

Instead of making the communication link carry the entire video stream, the airborne system performs the computationally expensive interpretation locally and communicates the resulting information.

This creates a clear research question:

> How much useful situational information can be extracted locally before communication is required?

That is a strong Edge AI systems question because it connects:

- AI inference;
- embedded computing;
- communication constraints;
- latency;
- power;
- privacy/security;
- human-machine interaction.

---

# 5. Why defence organisations could care

## 5.1 Communication-degraded environments

Defence systems cannot always assume reliable connectivity.

SENTINEL-X is designed around the assumption that communication may be:

- unavailable;
- intermittent;
- bandwidth-limited;
- unreliable;
- intentionally disrupted.

The architecture therefore remains useful even when the normal communication path is unavailable.

That is a meaningful design philosophy for tactical Edge computing.

---

## 5.2 Data sovereignty

The system does not depend on a cloud API.

The AI models, knowledge base and reasoning engine can remain locally deployed.

This means the demonstrator can investigate:

- offline AI;
- local knowledge retrieval;
- sensitive-data isolation;
- disconnected operation;
- model deployment on edge hardware.

For defence R&D, this is more relevant than a conventional cloud-based AI demonstration.

---

## 5.3 Low-bandwidth intelligence exchange

The project deliberately separates:

**raw sensor data**

from

**information that actually needs to reach the operator.**

The demonstrator can measure:

- original video bitrate;
- generated event size;
- number of events;
- end-to-end latency;
- bandwidth reduction;
- behaviour during communication interruptions.

This gives the project measurable engineering evidence instead of relying only on an architectural claim.

---

# 6. Independent AI verification

Another potentially attractive research element is the use of two heterogeneous computing platforms.

The Jetson proposes a detection.

The MCXN947 independently evaluates the candidate.

Conceptually:

```text
Model A
   ↓
Candidate
   ↓
Model B
   ↓
Verification
   ↓
Accepted / rejected
```

This is more interesting than simply saying that the project uses two AI models.

The research question becomes:

> Can heterogeneous, independently executed Edge inference improve the reliability of an embedded perception pipeline under constrained conditions?

That can be experimentally evaluated against a single-model baseline.

Important presentation point:

Do **not** claim that the architecture "eliminates false alarms" or "virtually eliminates false alarms" before experimental evidence exists.

Use:

> "designed to reduce false positives through independent verification."

Then measure the actual reduction.

---

# 7. Local multimodal RAG-VLM

The RAG-VLM layer moves the system beyond conventional object detection.

A normal CV system might output:

```text
Object: vehicle
Confidence: 0.91
Location: x,y
```

SENTINEL-X aims to provide additional context by combining:

- detected objects;
- spatial relationships;
- local maps;
- retrieved documents;
- domain knowledge;
- visual context.

The important research contribution is therefore not simply "using an LLM."

It is:

> **bringing multimodal contextual reasoning onto a disconnected Edge node.**

This is relevant to the broader development of Edge AI systems because the same architecture could be adapted to other high-consequence domains such as emergency response, infrastructure inspection, disaster response, and industrial monitoring.

---

# 8. Human-in-the-loop interface

The Raspberry Pi terminal gives the project another differentiating layer.

The operator can interact using:

```text
Speech
  ↓
Whisper.cpp
  ↓
Local SLM
  ↓
Local tactical state
  ↓
Piper
  ↓
Speech
```

This demonstrates local multimodal human-agent interaction.

For a defence R&D audience, the important point is not that the system can "talk."

The important point is that the operator can interact with locally generated information **without requiring cloud services or a conventional keyboard/display workflow**.

The system therefore becomes a human-machine teaming demonstrator.

---

# 9. What a defence engineer will see in the project

A strong technical evaluator could see evidence of experience in:

| Area | Demonstrated capability |
|---|---|
| Embedded AI | Deployment on Jetson + MCU |
| Edge CV | TensorRT / YOLO pipeline |
| TinyML | Lightweight CNN on MCXN947 |
| Heterogeneous computing | GPU + micro-NPU |
| Distributed systems | Airborne + ground nodes |
| GenAI | Local VLM/SLM |
| RAG | Offline knowledge retrieval |
| Multimodal AI | Vision + language + speech |
| Embedded security | TrustZone / secure execution concepts |
| Communications | Compact event-oriented telemetry |
| Human-machine interaction | STT + SLM + TTS |
| Systems engineering | End-to-end architecture |
| SWaP awareness | Power, latency and payload constraints |
| Validation | Quantitative benchmark potential |

This breadth is one of the project's strongest portfolio characteristics.

---

# 10. Why this is more valuable than a normal AI demo

A typical portfolio project might show:

```text
Camera → YOLO → Bounding Box
```

Another might show:

```text
Image → VLM → Text
```

Another might show:

```text
Documents → RAG → LLM
```

SENTINEL-X combines these into a constrained system:

```text
Sensor
 ↓
Edge perception
 ↓
Independent verification
 ↓
Local knowledge retrieval
 ↓
Multimodal reasoning
 ↓
Information compression
 ↓
Low-bandwidth communication
 ↓
Local human interaction
```

The engineering challenge is therefore in **integration and system behaviour**, not in claiming that any individual AI component is new.

---

# 11. What could make a defence person stop at the demonstration

The most convincing demonstration sequence would be:

### Step 1 — Show the live sensor

A scene is visible to the airborne Edge computer.

### Step 2 — Show local detection

The Jetson identifies an object.

### Step 3 — Show independent verification

The candidate is sent to the MCXN947.

The MCU independently evaluates it.

### Step 4 — Show contextual reasoning

The Jetson retrieves relevant local knowledge and generates a contextual event.

### Step 5 — Disconnect the network

The system continues operating.

This is important.

Do not merely say "offline."

Actually demonstrate:

```text
Internet: DISCONNECTED
Cloud:    NOT AVAILABLE
System:   STILL OPERATING
```

### Step 6 — Show the compact event

Display:

```text
Video stream:
Several Mbps

Tactical event:
<1 KB target
```

The actual measured numbers should be displayed rather than theoretical numbers.

### Step 7 — Demonstrate the operator loop

The Raspberry Pi receives the event.

The operator asks a question verbally.

Whisper.cpp transcribes it.

The local SLM processes it.

Piper responds.

That gives the audience a complete end-to-end story.

---

# 12. What will attract a project partner rather than just a visitor

A defence company is more likely to become interested when the project exposes a **clear integration opportunity**.

SENTINEL-X can deliberately expose interfaces for:

- camera/gimbal integration;
- different object-detection models;
- different MCU/NPU verification models;
- different VLMs;
- different local knowledge bases;
- different tactical radios;
- different ground terminals;
- different mapping systems;
- different UAV platforms.

This means a potential partner can look at the prototype and think:

> "We could replace this component with our hardware/product."

That is much more useful commercially than building a completely closed demonstration.

---

# 13. What partners could contribute

## UAV companies

Potential contribution:

- airframe;
- payload integration;
- flight testing;
- vibration testing;
- thermal/power integration;
- camera/gimbal integration.

## Sensor companies

Potential contribution:

- EO cameras;
- thermal cameras;
- multispectral sensors;
- stabilized gimbals.

## Defence communications companies

Potential contribution:

- low-bandwidth radio;
- MANET;
- resilient tactical networking;
- communication testing.

## Embedded semiconductor companies

Potential contribution:

- accelerator evaluation;
- NPU optimization;
- reference hardware;
- model deployment support.

## Defence research institutes

Potential contribution:

- datasets;
- adversarial robustness testing;
- evaluation methodology;
- field trials;
- human factors research.

## Universities / research groups

Potential contribution:

- formal verification;
- distributed Edge AI;
- multimodal reasoning;
- embedded security;
- human-agent teaming.

---

# 14. The project questions that can become research contributions

The strongest scientific direction is to turn the architecture into measurable experiments.

### Experiment 1 — Communication reduction

Compare:

```text
Continuous video transmission
vs.
Event-based semantic transmission
```

Measure:

- bandwidth;
- latency;
- energy;
- information retained.

### Experiment 2 — Independent verification

Compare:

```text
Jetson detector
vs.
Jetson + MCXN947 verification
```

Measure:

- precision;
- recall;
- false-positive rate;
- false-negative rate;
- latency;
- power.

### Experiment 3 — Offline reasoning

Compare:

```text
Cloud architecture
vs.
Fully local architecture
```

Measure:

- latency;
- availability;
- data exposure;
- bandwidth requirement.

### Experiment 4 — Communication interruption

Introduce:

```text
Normal connection
↓
Intermittent connection
↓
Complete disconnection
```

Measure what functions continue operating.

### Experiment 5 — Human interaction

Measure:

- speech-to-response latency;
- transcription accuracy;
- task completion time;
- operator interaction burden.

These experiments would turn the project from a collection of components into a serious systems research demonstrator.

---

# 15. What should NOT be claimed yet

The current project description contains several statements that should be treated as **targets**, not established facts.

For example:

- 60 FPS;
- 35 ms MCXN947 inference;
- <1 KB event;
- >99.9% RF reduction;
- >40% false-positive reduction;
- <2.5 s voice round trip;
- <280 g airborne payload;
- <25 W;
- <300 g;
- 100% air-gapped operation;
- specific model performance.

These should become:

> **Target metrics**

until measured on the actual hardware.

This will make the project much more credible to technically sophisticated defence engineers.

---

# 16. Important terminology correction

The project should avoid presenting itself as an autonomous system that independently decides military actions.

A safer and technically stronger framing is:

> **AI-assisted tactical reconnaissance and human-in-the-loop situational awareness.**

The AI:

- detects;
- verifies;
- retrieves;
- contextualizes;
- summarizes;
- highlights possible implications.

The human:

- evaluates;
- decides;
- acts.

This distinction also makes the system easier to position as an R&D demonstrator.

---

# 17. What makes the project attractive at an expo

At an expo, most people will not read the entire architecture.

The first 30 seconds should communicate:

> **"The drone does not need to continuously send its video to the operator. It understands the scene locally, verifies important detections on a second processor, retrieves relevant local knowledge, and sends only compact intelligence to the ground terminal. The entire AI pipeline works without cloud connectivity."**

Then demonstrate it.

That is much easier to understand than starting with model names.

---

# 18. Recommended demonstration dashboard

The screen should show four panels:

```text
┌─────────────────────────────────────────────────────────┐
│ LIVE EDGE PERCEPTION                                    │
│ Camera → YOLO → Tracking                                │
├──────────────────────────┬──────────────────────────────┤
│ MCU VERIFICATION         │ LOCAL RAG-VLM                │
│ Jetson:   DETECTED       │ Retrieved knowledge: 3       │
│ MCXN947:  VERIFIED       │ Context analysis: ...       │
├──────────────────────────┴──────────────────────────────┤
│ COMMUNICATION                                            │
│ Video stream:      X Mbps                               │
│ Event message:     X bytes                              │
│ Network:           DISCONNECTED                         │
├─────────────────────────────────────────────────────────┤
│ OPERATOR                                                │
│ Speech → Whisper → SLM → Piper                         │
└─────────────────────────────────────────────────────────┘
```

This makes the distributed architecture visually obvious.

---

# 19. What a partner should be able to ask

A technically interested person will probably ask:

1. What happens when the communication link disappears?
2. How much bandwidth does the system actually save?
3. How independent is the second inference model?
4. Why use a microcontroller instead of another GPU?
5. How much power does each node consume?
6. What is the end-to-end latency?
7. Can the models be replaced?
8. Can the system run completely offline?
9. How is the local knowledge base updated?
10. How robust is the perception system to environmental changes?
11. What happens when the two models disagree?
12. Can the architecture be integrated into another UAV?
13. What are the actual measured benchmarks?
14. How does the system behave under intermittent communication?

Preparing strong measured answers to these questions will make the project considerably more credible.

---

# 20. The strongest message for defence partners

The project should not be sold as:

> "I built a drone AI."

It should be presented as:

> **"I built a distributed Edge AI architecture for operating when connectivity cannot be assumed."**

The drone is simply the demonstration platform.

The deeper technology is:

```text
Heterogeneous Edge Computing
+
Independent AI Verification
+
Local Multimodal RAG
+
Disconnected Operation
+
Low-Bandwidth Semantic Communication
+
Human-Agent Interaction
```

That is the part that can attract research and engineering discussions.

---

# 21. Overall assessment

SENTINEL-X has several characteristics that can make it interesting to defence R&D people:

### Technical depth

It combines embedded AI, TinyML, GPU inference, NPU inference, multimodal AI, RAG, distributed systems, communications and human-machine interaction.

### Demonstrability

The architecture can be demonstrated with commercially available development hardware.

### Integration potential

Individual components can potentially be replaced or integrated with partner hardware.

### Research potential

The architecture produces measurable questions around reliability, bandwidth, latency, power and human interaction.

### Defence relevance

Disconnected operation, local processing, communication efficiency and situational awareness are directly relevant to tactical Edge computing.

### Portfolio value

A functioning prototype would demonstrate system-level Edge AI engineering rather than only model training.

---

# 22. Final positioning

The strongest positioning is:

> **SENTINEL-X is a research demonstrator for heterogeneous, air-gapped Edge AI that transforms locally observed sensor data into compact, context-rich information for human operators under communication-constrained conditions.**

The project's value is not that YOLO, RAG, VLM, Whisper, or TinyML are individually new.

Its value is the **system-level integration, constrained hardware implementation, and measurable behaviour of the complete distributed architecture**.

For a defence R&D audience, the most important next step is therefore not adding more AI components.

It is proving the existing architecture with measurements:

**latency + power + bandwidth + reliability + verification accuracy + offline operation.**

Those measurements are what can turn the project from an impressive concept into a credible defence technology demonstrator.
