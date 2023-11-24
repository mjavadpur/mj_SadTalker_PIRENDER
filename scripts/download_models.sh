mkdir ./checkpoints  

# lagency download link
# wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/auido2exp_00300-model.pth -O ./checkpoints/auido2exp_00300-model.pth
# wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/auido2pose_00140-model.pth -O ./checkpoints/auido2pose_00140-model.pth
# wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/epoch_20.pth -O ./checkpoints/epoch_20.pth
# wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/facevid2vid_00189-model.pth.tar -O ./checkpoints/facevid2vid_00189-model.pth.tar
# wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/shape_predictor_68_face_landmarks.dat -O ./checkpoints/shape_predictor_68_face_landmarks.dat
# wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/wav2lip.pth -O ./checkpoints/wav2lip.pth
# wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/mapping_00229-model.pth.tar -O ./checkpoints/mapping_00229-model.pth.tar
# wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/mapping_00109-model.pth.tar -O ./checkpoints/mapping_00109-model.pth.tar
# wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/hub.zip -O ./checkpoints/hub.zip
# unzip -n ./checkpoints/hub.zip -d ./checkpoints/

gdown https://drive.google.com/uc?id=1-0xOf6g58OmtKtEWJlU3VlnfRqPN9Uq7 
unzip -x ./face.zip
mv /content/mj_SadTalker_pirender/face/latest_checkpoint.txt /content/mj_SadTalker_pirender/checkpoints
mv /content/mj_SadTalker_pirender/face/epoch_00190_iteration_000400000_checkpoint.pt /content/mj_SadTalker_pirender/checkpoints
# rm face.zip

#### download the new links.
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar -O  ./checkpoints/mapping_00109-model.pth.tar
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar -O  ./checkpoints/mapping_00229-model.pth.tar
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors -O  ./checkpoints/SadTalker_V0.0.2_256.safetensors
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors -O  ./checkpoints/SadTalker_V0.0.2_512.safetensors
wget -nc https://huggingface.co/vinthony/SadTalker-V002rc/blob/main/epoch_00190_iteration_000400000_checkpoint.pt -O ./checkpoints/epoch_00190_iteration_000400000_checkpoint.pt 


# wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip -O ./checkpoints/BFM_Fitting.zip
# unzip -n ./checkpoints/BFM_Fitting.zip -d ./checkpoints/

### enhancer 
mkdir -p ./gfpgan/weights
wget -nc https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth -O ./gfpgan/weights/alignment_WFLW_4HG.pth 
wget -nc https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -O ./gfpgan/weights/detection_Resnet50_Final.pth 
wget -nc https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -O ./gfpgan/weights/GFPGANv1.4.pth 
wget -nc https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -O ./gfpgan/weights/parsing_parsenet.pth 

