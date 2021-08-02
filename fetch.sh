#sync the files to a local folder with selected files only
mkdir dataset
aws s3 sync s3://avspeechesrgandataset/ /home/ubuntu/Wav2Lip/dataset/ --exclude '*' --include '*audio.wav'  --include '*/align/*'

#refactor data for Wav2Lip training
rm -rf preprocessed_avspeech
mkdir preprocessed_avspeech
python3 refactory_dataset.py

#create filelist

