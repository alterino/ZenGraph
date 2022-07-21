import argparse
import openai

from test_samples import samples

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default=None)

models = ['ada:ft-personal-2022-07-13-01-38-40', 
          'babbage:ft-personal-2022-07-13-01-18-54',
          'curie:ft-personal-2022-07-13-00-51-02',
          'davinci:ft-personal-2022-07-13-01-46-19'] 

for m in models:
    for prompt in samples:
        openai.Completion.create(
                model = m,
                prompt = prompt)


breakpoint()



