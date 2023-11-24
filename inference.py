from argparse import ArgumentParser
import time
import torch, uuid
import os, sys, shutil, platform
from src.facerender.pirender_animate import AnimateFromCoeff_PIRender
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data

from src.utils.init_path import init_path

from pydub import AudioSegment


def mp3_to_wav(mp3_filename,wav_filename,frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename,format="wav")




def main(args):
    # , source_image, driven_audio, preprocess='crop', 
    #     still_mode=False,  use_enhancer=False, batch_size=1, size=256, 
    #     pose_style = 0, 
    #     facerender='facevid2vid',
    #     exp_scale=1.0, 
    #     use_ref_video = False,
    #     ref_video = None,
    #     ref_info = None,
    #     use_idle_mode = False,
    #     length_of_audio = 0, use_blink=True,
    #     result_dir='./results/'):
    inference_startTime = time.time()
    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, time.strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose
    use_idle_mode = False # args.use_idle_mode
    length_of_audio = args.length_of_audio
    use_ref_video = False # args.use_ref_video
    ref_info = args.ref_info
    ref_video = args.ref_video
    facerender = args.facerender
    
    current_root_path = os.path.split(sys.argv[0])[0]
    os.environ['TORCH_HOME']= args.checkpoint_dir
    config_path = 'src/config'

    init_path_startTime = time.time()

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, config_path), args.size, args.old_version, args.preprocess)
    print(sadtalker_paths)
        
    audio_to_coeff = Audio2Coeff(sadtalker_paths, args.device)
    preprocess_model = CropAndExtract(sadtalker_paths, args.device)
    
    if facerender == 'facevid2vid' and args.device != 'mps':
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, args.device)
    elif facerender == 'pirender' or args.device == 'mps':
        animate_from_coeff = AnimateFromCoeff_PIRender(sadtalker_paths, args.device)
        facerender = 'pirender'
    else:
        raise(RuntimeError('Unknown model: {}'.format(facerender)))
        

    print("save_dir:" + save_dir)
    input_dir = os.path.join(save_dir, 'input')
    os.makedirs(input_dir, exist_ok=True)

    print("input_dir:" + input_dir)
    print("pic_path:" + pic_path)
    # pic_path = os.path.join(input_dir, os.path.basename(pic_path)) 
    # shutil.move(pic_path, input_dir)

    if audio_path is not None and os.path.isfile(audio_path):
        # audio_path = os.path.join(input_dir, os.path.basename(audio_path))  

        #### mp3 to wav
        if '.mp3' in audio_path:
            mp3_to_wav(audio_path, audio_path.replace('.mp3', '.wav'), 16000)
            audio_path = audio_path.replace('.mp3', '.wav')
        else:
            print("audio_path:"+audio_path)
            shutil.copy(audio_path, input_dir)

    elif use_idle_mode:
        audio_path = os.path.join(input_dir, 'idlemode_'+str(length_of_audio)+'.wav') ## generate audio from this new audio_path
        from pydub import AudioSegment
        one_sec_segment = AudioSegment.silent(duration=1000*length_of_audio)  #duration in milliseconds
        one_sec_segment.export(audio_path, format="wav")
    else:
        print(use_ref_video, ref_info)
        # assert use_ref_video == True and ref_info == 'all'

    if use_ref_video and ref_info == 'all': # full ref mode
        ref_video_videoname = os.path.basename(ref_video)
        audio_path = os.path.join(save_dir, ref_video_videoname+'.wav')
        print('new audiopath:',audio_path)
        # if ref_video contains audio, set the audio from ref_video.
        cmd = r"ffmpeg -y -hide_banner -loglevel error -i %s %s"%(ref_video, audio_path)
        os.system(cmd)        

    os.makedirs(save_dir, exist_ok=True)
    
    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,\
                                                                            source_image_flag=True, pic_size=args.size)
    
    if first_coeff_path is None:
        raise AttributeError("No face is detected")

    if use_ref_video:
        print('using ref video for genreation')
        ref_video_videoname = os.path.splitext(os.path.split(ref_video)[-1])[0]
        ref_video_frame_dir = os.path.join(save_dir, ref_video_videoname)
        os.makedirs(ref_video_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing pose')
        ref_video_coeff_path, _, _ =  preprocess_model.generate(ref_video, ref_video_frame_dir, args.preprocess,\
                                                                            source_image_flag=True)
    else:
        ref_video_coeff_path = None

    if use_ref_video:
        if ref_info == 'pose':
            ref_pose_coeff_path = ref_video_coeff_path
            ref_eyeblink_coeff_path = None
        elif ref_info == 'blink':
            ref_pose_coeff_path = None
            ref_eyeblink_coeff_path = ref_video_coeff_path
        elif ref_info == 'pose+blink':
            ref_pose_coeff_path = ref_video_coeff_path
            ref_eyeblink_coeff_path = ref_video_coeff_path
        elif ref_info == 'all':            
            ref_pose_coeff_path = None
            ref_eyeblink_coeff_path = None
        else:
            raise('error in refinfo')
    else:
        ref_pose_coeff_path = None
        ref_eyeblink_coeff_path = None

    #audio2ceoff
    if use_ref_video and ref_info == 'all':
        coeff_path = ref_video_coeff_path # audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
    else:
        batch = get_data(first_coeff_path, audio_path, args.device, ref_eyeblink_coeff_path=ref_eyeblink_coeff_path, still=args.still, \
            idlemode=use_idle_mode, length_of_audio=length_of_audio, use_blink=args.use_blink) # longer audio?
        coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, still_mode=args.still, \
        preprocess=args.preprocess, size=args.size, expression_scale = args.expression_scale, facemodel=facerender)
    return_path = animate_from_coeff.generate(data, save_dir,  pic_path, crop_info, enhancer=args.enhancer, preprocess=args.preprocess, img_size=args.size)
    video_name = data['video_name']
    print(f'The generated video is named {video_name} in {save_dir}')
    shutil.move(return_path, save_dir+'.mp4')

    del preprocess_model
    del audio_to_coeff
    del animate_from_coeff

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    import gc; gc.collect()
    
    
    inference_endTime = time.time()
    
    execution_time = inference_endTime - inference_startTime
    print(f"Inference Execution Time: {execution_time} seconds")

if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 
    parser.add_argument("--use_idle_mode",default=False, action="store_false", help="use_idle_mode" ) 
    parser.add_argument("--length_of_audio", type=int, default=0,  help="length_of_audio")
    parser.add_argument("--use_ref_video",default=False,action="store_false", help="use_ref_video" ) 
    parser.add_argument("--ref_info", default=None, choices=['pose', 'blink', 'pose+blink', 'all'], help="" ) 
    parser.add_argument("--ref_video", default=None, help="ref_video")
    parser.add_argument("--use_blink",action="store_true", help="use_blink" ) 
    parser.add_argument('--facerender',  type=str, default='pirender', help="Face enhancer, [facevid2vid, pirender]")
 

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)

  