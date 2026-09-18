Should v2 replace the baseline?
No
What did overall accuracy suggest, and what did the slices and the gate say?

Which single check decided it for you?
If you would not ship as-is, what is one mitigation? (Gather training data for the weak slice, keep the baseline on that slice, batch requests, quantize or distill to a smaller model, raise the latency budget with sign-off, …)
Your final recommendation.

# how to improve latency of v2?

- distillation
- quantization: Artificial Intelligence & LLMs: Shrinks model sizes (like changing 32-bit floating-point numbers to 4-bit or 8-bit integers) so large AI models can run on personal computers or phones with less memory.


Distillation and quantization are both AI model compression techniques, but quantization shrinks the numeric precision of an existing model's weights, while distillation trains a brand-new, smaller student model to mimic a larger teacher model.

QuantizationHow it works: Keeps the exact same model structure, but stores numbers using fewer bits (such as dropping from 16-bit floats down to 8-bit or 4-bit integers).Training requirement: Zero or minimal retraining needed.Pros: Fast and simple to implement; dramatically cuts memory usage with minimal accuracy loss.Cons: Limited by the original parameter count; very low-bit widths can hurt performance.Best used for: A quick, straightforward reduction in memory and hardware acceleration needs.

DistillationHow it works: Uses a large "teacher" model to teach a separate, smaller "student" model by transferring knowledge and soft target probabilities.Training requirement: Demands a heavy, time-intensive training run from scratch.Pros: Creates an ultra-compact, customized architecture that can run efficiently on tiny edge devices.Cons: Expensive and resource-heavy training process.Best used for: Achieving maximum physical size reduction for specialized deployments
