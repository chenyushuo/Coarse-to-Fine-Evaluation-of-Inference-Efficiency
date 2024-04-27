import torch
from torch.profiler import profile, record_function, ProfilerActivity
import sys
import transformers
from vllm import LLM, SamplingParams
import pickle

# torch.set_num_threads(1)


prompt = "Please complete the following sentences:\n\nYou wanted to know more about greenwashing, and Scot Case, from environmental marketing firm TerraChoice, answered. Greenwashing expert Scot Case of TerraChoice. \"Why are green products often more expensive than ones that don't say they are green or environmentally friendly? Is it just because green has become a new form of 'premium brand'? Isn't this bad news if we want to make more people environmentally aware when they go shopping?\" Harriet Gladwell. Case: First, it should be noted that not all greener products are more expensive. The remanufactured toner cartridges I purchase at a nationwide office-supply store, for example, carry the same warranty as other cartridges at a 30-percent lower cost. This greener option is less expensive because the manufacturer avoids the cost of manufacturing the plastic and electronic components. They simply reuse the parts from recycled cartridges. There are also greener products that do not cost extra. There are cleaning products and paints, for example, that have been certified as meeting tough environmental standards by EcoLogo or Green Seal that deliver the same high-quality performance one expects without costing any extra. Other greener products might be slightly more expensive initially, but generate substantial savings for the consumer. Energy-efficient compact fluorescent lightbulbs (CFLs), for example, are still four times more expensive than traditional cheap incandescent light bulbs. However, CFLs use 75 percent less electricity and last 10 times longer, so they don't have to be replaced as frequently. As a result, the typical CFL saves consumers $30 over the life of the bulb. There are now energy- or water-efficient versions of all sorts of products -- refrigerators, windows, air conditioners, televisions, dishwashers, ovens, showerheads, washing machines, etc. The more efficient versions are typically more expensive initially to reflect the higher-quality components used to make them, but they quickly pay for themselves in lower energy and water costs. Look for products that are Energy Star registered. Even better, look for products that have been independently certified as meeting the Energy Star standards. Why are other greener products still more expensive sometimes? It boils down to the simple laws of supply and demand. Any new innovative product, whether it is \"greener\" or not, costs extra initially. It costs money to research and develop the product and to build the factories and supply chains it takes to make the product. Manufacturers try to recoup those costs as quickly as possible during the initial sales of the product. As demand increases, however, additional manufacturing efficiencies -- economies of scale -- begin to emerge that permit the prices to fall. In addition, high prices attract competitors with similar products, and the additional competition helps force prices lower. Are some manufacturers attempting to earn additional revenue by presenting their greener options as a premium brand? Absolutely. Just as some clothing manufacturers charge extra to have their name brand applied to a shirt. It is also possible, however, to buy high-quality, greener products, at very good prices, at growing numbers of mainstream retail outlets. When DVD players and cell phones were first introduced, they were only available to the very wealthy. Now everyone has at least one. The same is increasingly true with greener product offerings. \"What are the most obvious signs that a company is greenwashing the public with false claims? What words and phrases should raise a red flag?\" Carla Dos Santos. The most obvious sign a company is greenwashing is if the company fails to provide proof of their environmental claims. Legitimate environmental claims can be certified by independent outside third-party auditors. Manufacturers can also provide test data and other relevant information on Web sites. Consumers should also beware of generic environmental claims that are so vague they are likely to be misunderstood. Watch out for broad claims like \"eco-friendly,\" \"earth kind,\" \"all natural,\" \"eco-safe\" or other green babble. Even phrases like \"biodegradable,\" \"recyclable\" and \"compostable\" can be misleading if they fail to clarify how the products were tested or under what circumstances the claim is true. Make sure any environmental claim is specific, backed by proof, and, preferably, verified by an independent, outside third-party. For additional greenwashing examples and recommendations on how to avoid being foo"
prompt = (prompt+" ")*5

tokenizer = transformers.AutoTokenizer.from_pretrained("/run/user/1000/Llama-2-7b-chat-hf", padding_side="left")
tokens = tokenizer.encode(prompt)
tokens = tokens[:int(sys.argv[2])]
prompt = tokenizer.decode(tokens)


prompts = [prompt for _ in range(int(sys.argv[1]))]

sampling_params = SamplingParams(temperature=0, max_tokens=2)

# llm = LLM(model="/run/user/1000/Llama-2-7b-chat-hf", swap_space=0, seed=42,enforce_eager=True,max_num_batched_tokens=32768)

llm = LLM(model="/run/user/1000/Llama-2-7b-chat-hf", swap_space=0, seed=42,enforce_eager=True)
torch.cuda.cudart().cudaProfilerStart()

# torch.cuda.nvtx.range_push("114")
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, profile_memory=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
outputs = llm.generate(prompts, sampling_params)
torch.cuda.cudart().cudaProfilerStop()

# with open(f"./tests_" + sys.argv[1] +"_" +sys.argv[2]+ "_3090"  +".pkl" , 'wb') as f:
#     pickle.dump(prof.profiler.function_events, f)
print(len(tokens))

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# torch.cuda.nvtx.range_pop()

# torch.cuda.cudart().cudaProfilerStop()


