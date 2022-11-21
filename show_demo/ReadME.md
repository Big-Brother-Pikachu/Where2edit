The Conda environment is in "requirements.txt" in the main folder. And you need three public pre-trained models in the "pretrained_models" folder in the main folder, which are:

The pre-trained StyleGAN2 is from <https://github.com/rosinality/stylegan2-pytorch>.

The pre-trained CLIP is from <https://github.com/openai/CLIP>.

The GAN inversion model is from <https://github.com/omertov/encoder4editing>.

Our trained model can be dowmloaded from <https://cloud.tsinghua.edu.cn/d/8a65168fa4464014b62f/>.
You have to put it (the "final_mapper.pt") in the "models" folder. 
Then you can run the demo in this folder with: `streamlit run try_demo.py --server.port YOUR_PORT`

Have fun editing!