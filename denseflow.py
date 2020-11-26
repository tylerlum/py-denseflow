import os
import numpy as np
import cv2
from PIL import Image
import argparse

ABS_PATH_TO_THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def ToImg(raw_flow, bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow = raw_flow
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow += bound
    flow *= (255 / float(2 * bound))
    return flow


def save_flows(flows, image, save_dir, num, bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: no returns
    '''
    # rescale to 0~255 with the bound setting
    flow_x = ToImg(flows[..., 0], bound)
    flow_y = ToImg(flows[..., 1], bound)
    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(os.path.join(save_dir))

    # save the image
    save_img = os.path.join(save_dir, 'img_{:05d}.jpg'.format(num))
    cv2.imwrite(save_img, image)

    # save the flows
    save_x = os.path.join(save_dir, 'flow_x_{:05d}.jpg'.format(num))
    save_y = os.path.join(save_dir, 'flow_y_{:05d}.jpg'.format(num))
    cv2.imwrite(save_x, flow_x)
    cv2.imwrite(save_y, flow_y)


def dense_flow(abs_path_to_video, abs_path_to_output, step, bound):
    '''
    To extract dense_flow images
    :param abs_path_to_video: absolute path to vidoe file to process
    :param abs_path_to_output: absolute path to where outputs should be saved
    :param step: num of frames between each two extracted frames
    :param bound: bi-bound parameter
    :return: no returns
    '''
    # Setup optical flow calculator
    dtvl1 = cv2.optflow.createOptFlow_DualTVL1()

    # Setup video capture
    video_capture = cv2.VideoCapture(abs_path_to_video)
    if not video_capture.isOpened():
        print(f'Could not initialize capturing for {abs_path_to_video}')
        exit()
    total_num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    image, prev_image, gray, prev_gray = None, None, None, None
    num_frames_flow_processed, num_frames_seen = 0, 0

    while True:
        # Get next frame
        valid, frame = video_capture.read()
        if not valid:
            print(f"Done processing {abs_path_to_video}")
            break
        num_frames_seen += 1
        print(f"On frame {num_frames_seen} / {total_num_frames}. Flow processed: {num_frames_flow_processed}")

        # Resize for consistent resolution
        frame = cv2.resize(frame, (224, 224))

        # Handle first loop
        if num_frames_flow_processed == 0:
            # Store first frame
            prev_image = frame
            prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
            num_frames_flow_processed += 1

            # Move forward step-1 frames
            for _ in range(step - 1):
                _ = video_capture.read()
                num_frames_seen += 1

        # Regular case
        else:
            image = frame
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Calculate optical flow using dual tvl1 algorithm
            frame_0 = prev_gray
            frame_1 = gray
            flowDTVL1 = dtvl1.calc(frame_0, frame_1, None)

            # Save flow as image
            save_flows(flowDTVL1, image, abs_path_to_output, num_frames_flow_processed, bound)

            # Update values for next loop
            prev_gray = gray
            prev_image = image
            num_frames_flow_processed += 1

            # Move forward step-1 frames
            for _ in range(step - 1):
                _ = video_capture.read()
                num_frames_seen += 1


def parse_args():
    # <input_videos_root>  # Expects videos
    # └── Cov_video1.mp4
    # └── Cov_video2.mp4
    # └── Pne_video1.mp4
    # └── Pne_video1.mp4
    # └── Reg_video1.mp4
    # └── Reg_video1.mp4

    # <output_root>  # Output dir is created
    # └── Cov_video1
    #         └── flow_x_00001.jpg
    #         └── flow_y_00001.jpg
    #         └── img_00001.jpg
    #     ...
    # └── Cov_video2
    #         └── flow_x_00001.jpg
    #         └── flow_y_00001.jpg
    #         └── img_00001.jpg
    #     ...
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--input_videos_root', default=os.path.join(ABS_PATH_TO_THIS_FILE_DIR, 'videos'), type=str)
    parser.add_argument('--output_root', default=os.path.join(ABS_PATH_TO_THIS_FILE_DIR, 'flows'), type=str)
    parser.add_argument('--step', default=1, type=int, help='gap between optical flow frames')
    parser.add_argument('--bound', default=15, type=int, help='maximum absolute optical flow value')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Specify the augments
    input_videos_root = args.input_videos_root
    output_root = args.output_root
    step = args.step
    bound = args.bound

    # Get video list
    video_list = [video for video in os.listdir(input_videos_root)]

    print(f"Found {len(video_list)} videos")
    output_dirs = [video.split('.')[0] for video in video_list]

    # Dense flow for all videos
    for video, output_dir in zip(video_list, output_dirs):
        print(f"Processing {video}")
        print("=============================================")
        abs_path_to_video = os.path.join(input_videos_root, video)
        abs_path_to_output = os.path.join(output_root, output_dir)
        dense_flow(abs_path_to_video, abs_path_to_output, step, bound)