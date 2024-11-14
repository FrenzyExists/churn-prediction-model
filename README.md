# Churn Prediction Model

A project made part of the headstarter fellowship program

took around 15 hours or so

project is broken in two parts, the machine learning model and the frontend ui. The frontend is made with streamit.


Some thoughts

Streamit seems like a much better alternative to a library used before, dash. The downside I can think of right now is the fact that the graphs are not really that interactive compared to what Plotly offers. That and the fact that for complex applications that may require custom components and the such, using the js framework of plotly and combining that with EEl is still a better approach for maximum customizability. Streamit is still pretty good tho, it really abstract all the async calls and the things done in react. I somewhat wish that dash could do this, even with some of the UI kits available for dash it feels like a hassle to do things there than to separate the the python backend layer from the frontend UI layer altogether. Yeah, I like Streamit