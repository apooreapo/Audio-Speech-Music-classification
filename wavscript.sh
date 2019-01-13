#!/bin/sh

for fileName in *
do
	ffmpeg -ss -i "$fileName" -acodec pcm_s16le -ac 1 -ar 22050 copy"${fileName}"
done
