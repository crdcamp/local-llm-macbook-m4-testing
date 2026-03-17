# The Proper Way

So most of this was just introductory stuff / playing around with code. However... there's definitely some better ways to do all of this. First of all...

## curl for pulling models

I believe you can download these models using `curl` or something similar. Python might actually be completely unnecessary for the majority of this project.

## llama-bench for inspecting models

The whole python investigation thing with the charts and shit was mostly just for fun and Python practice. In reality, `llama-bench` can handle (I'm assuming) all of this for us. However, if you ever want to pinpoint the performance more closely to your liking (since `llama-bench` doesn't allow for custom test prompts) then you can start thinking about researching into ways to determine the computational performance of a local model.
