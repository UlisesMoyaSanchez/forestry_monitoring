In particular, we would be grateful if you could develop an interactive Python notebook that helps participants gain practical experience with AI applications for forestry monitoring using satellite imagery and/or remote sensing data. The tutorial should provide a clear workflow that guides learners through the relevant climate and forestry context, data inputs, modeling or analytical steps, interpretation of results, and important limitations or responsible-use considerations. The goal is for participants to leave with both a conceptual understanding of the topic and practical experience they can build on after the program.

Tutorials will be completed by participants asynchronously during the program and accompanied by walkthrough videos. Examples of previous code notebook tutorials, which may be useful for understanding the expected length and format, can be found here; examples of past walkthrough videos can also be found here. Once you confirm your participation, we will share more detailed guidance on the tutorial format, scope, and review process.

By accepting this invitation, you would be committing to:

    Preparing an interactive Python notebook tutorial before the start of the Virtual Summer School.

    Tutorial creators will be asked to deliver the following:
Interactive Python notebook: A code notebook tutorial following the CCAI Summer School 2026 tutorial notebook template. This should be accompanied by:
requirements.txt file stating all dependencies and versions for software, packages, or tooling.
Datasheets and/or model cards where applicable.
Walkthrough video: A short walkthrough video to support self-paced learning.
Quiz questions: 3-4 multiple choice or true/false quiz questions that can be used to assess participants’ comprehension of key learning objectives.

CCAI Virtual Summer School | Tutorial Goals and Format

Tutorials play an important role in the CCAI Summer School, providing participants hands-on practice with employing AI and machine learning (ML) techniques for climate-relevant problems. They are where participants move from understanding concepts to actually applying them.

Contents of this document:
Collective learning objectives for tutorials
Tutorial-specific learning objectives
Tutorial audience and difficulty level
Tutorial format
Collective learning objectives for tutorials
In support of the overall program learning objectives, the goal is that tutorials will collectively enable participants to:
Gain exposure to impactful AI-for-climate applications spanning different sectors, disciplines, and AI/ML approaches.
Understand concrete considerations for developing and actualizing impactful AI-for-climate work, including those related to problem framing, data collection, algorithm selection (including what requirements algorithms must satisfy), metrics and evaluation, and/or deployment.
Gain hands-on practice in implementing AI/ML pipelines – including task definition, working with and exploring data, model development/training, and evaluation – in the context of a climate-relevant problem.
Understand important technical considerations within the AI/ML pipeline to improve the model’s performance and fit to the task (e.g., considerations for data selection and transformation, model design, preventing over-/underfitting, troubleshooting failures).
Gain familiarity with technical best practices for documentation and measurement, such as datasheets, model cards, and energy/emissions measurement.
Understand important ethical considerations for developing and actualizing AI-for-climate work, such as potential pitfalls; considerations associated with fairness, accountability, transparency, and equity; connections to responsible AI and climate justice; potential side effects and unintended consequences; etc.

Tutorial-specific learning objectives
Each tutorial within the 2026 CCAI Virtual Summer School should give participants a clear, practical, and self-contained learning experience that connects AI methods to real climate-relevant problems. The goal is not only to show how an AI/ML method works – each tutorial should help participants understand:
What climate-relevant AI/ML problem is being addressed
Why the problem matters, and its connection to climate impact
What data, model, tool, simulator, or workflow is being used
How the method is applied in practice
How to interpret the results
What the method can and cannot tell us
Limitations or responsible-use considerations
The pathway to deployment and pathway to impact.

Each tutorial should focus on one clear task, workflow, dataset, method, tool, or decision context. It should not try to cover an entire sector or research field. For example:

Too broad
Stronger focus
Teach AI for hydrology
Use time-series data to forecast streamflow and discuss how forecasts support water management
Teach AI for agriculture
Use climate and crop data to build a crop-yield prediction workflow
Teach remote sensing for climate
Use satellite imagery to detect deforestation and discuss limitations
Teach climate risk modeling 
Build an interpretable climate-risk score using exposure and vulnerability indicators


Each tutorial should define clear, specific learning objectives outlining what participants are expected to gain or understand by the end of the tutorial. Learning objectives should be written in a simple and direct way and included within the tutorial notebook, so that participants can easily understand what they will get out of the tutorial. Making this explicit also helps ensure that the content, structure, and exercises are all aligned with the intended outcomes.

Overall, tutorials should not feel like standalone exercises. They should connect clearly to the broader program and contribute to a consistent learning experience across modules.
Tutorial audience and difficulty level
The Summer School audience is broad and global, including participants from AI/ML, climate-relevant fields, policy, civil society, the private sector, and academia. Participants will have different levels of Python, AI/ML, and climate experience, so tutorials should use clear explanations, define key terms, and avoid assuming too much background knowledge.

Each tutorial should be accessible to participants with a basic understanding of AI/ML and climate. Note, however, that this does not mean that your tutorial must shy away from advanced concepts methods – notably, some participants coming from advanced backgrounds will be eager to learn about these! This simply means that where advanced concepts are presented, either (a) there should be enough explanation that entry-level participants are able to follow the general idea of what is being presented, or (b) this content should be clearly marked as optional and should not be required to complete the core tutorial.

We will provide participants with some pre-program materials, such as publicly available Python programming resources and open ML MOOCs. In order to receive a certificate of attendance for the Virtual Summer School program, all participants will also be required to take either the Introduction to AI or Introduction to Climate Change module (depending on their background) as part of the Virtual Summer School. If (parts of) your tutorial assume more substantial prior knowledge of AI/ML than described here, please flag those prerequisites within the tutorial itself. 
Tutorial format
Tutorials should follow the format presented in the CCAI Summer School 2026 Tutorial Notebook Template. 
Examples of previous code notebook tutorials, which may be useful for understanding the expected length and format, can be found here. Examples of past walkthrough videos can also be found here.
Length: The tutorial will be completed asynchronously by participants. We would like to limit the time that students spend on this tutorial to 2 hours.
Easy-to-run: As the program is largely asynchronous and delivered at scale, tutorials need to be:
Clear and self-contained
Easy to follow without live support
Designed for participants working at their own pace 
Please test the tutorial from start to finish in a fresh environment before submission. If a dataset is large, restricted, or slow to access, provide a smaller sample dataset or fallback option.
Interactivity: Participants should not only run completed code cells. Tutorials should include a few purposeful interactive or reflective elements, such as:
Asking participants to change a parameter and to note the effects on the model
Challenging the participant to improve the performance of the model and providing some possible strategies
Answer interpretation questions or reflect on whether the output is sufficient for a real-world decision.
Real-world data conditions: Tutorials should reflect real-world data conditions. Climate-relevant data may be incomplete, noisy, biased, unevenly distributed, delayed, or difficult to access. Each tutorial should briefly explain what dataset is being used, where it comes from, what it represents, key variables or inputs, important limitations or gaps, and any preprocessing or simplifications made for teaching purposes.
Responsible-use considerations throughout: Responsible-use considerations should be integrated throughout the tutorial, not added only as a final note. Each tutorial should help participants understand what the workflow is intended to support, what it should not be used for, who could be affected if the output is wrong, and what validation, expertise, or governance would be needed before real-world use.
Energy/emissions measurement: As a demonstration of best practice, please also use the CodeCarbon library to instrument your code for energy and emissions tracking. See this tutorial for example usage of CodeCarbon within a Colab notebook.




    
