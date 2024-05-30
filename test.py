from haystack.components.generators import HuggingFaceAPIGenerator
import os
from getpass import getpass
from haystack.utils import Secret

os.environ["HF_API_TOKEN"] = getpass("Your HuggingFace Hub token:")


llm = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                              api_params={"model": "meta-llama/Meta-Llama-3-8B-Instruct"})
# token=Secret.from_token("YOUR_HF_API_TOKEN"
# ))
result = llm.run(prompt="""
  Would you be willing to financially back an inventor who is marketing a device
  that she claims has 25 kJ of heat transfer at 600 K, has heat transfer to the
  environment at 300 K, and does 12 kJ of work? Explain your answer by examining
  the thermodynamic properties of this device, but be concise.
""")

# Solution:
# The heat transfer to the cold reservoir is Qc=Qh−W=25kJ−12kJ=13kJ, so the
# efficiency is Eff=1−QcQh=1−13kJ25kJ=0.48. The Carnot efficiency is
# EffC=1−TcTh=1−300K600K=0.50. The actual efficiency is 96% of the Carnot
# efficiency, which is much higher than the best-ever achieved of about 70%,
# so her scheme is likely to be fraudulent.
# From https://phys.libretexts.org/Bookshelves/University_Physics/Exercises_(University_Physics)/Exercises%3A_College_Physics_(OpenStax)/15%3A_Thermodynamics_(Exercises)

print(result)
