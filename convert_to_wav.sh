#! /bin/zsh

if [ $# != 1 ]; then
    echo "usage: $0 DATADIR" 1>&2
    exit 0
fi

for mp3file in `\find $1 -maxdepth 2 -type f | grep .mp3`; do
    ffmpeg -i $mp3file -ac 1 -ar 44100 -acodec pcm_s16le ${mp3file/mp3/wav}
done
