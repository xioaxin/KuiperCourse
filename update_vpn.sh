#!/bin/bash
curl 'https://glados.rocks/api/user/checkin' \
      -H 'authority: glados.rocks' \
      -H 'accept: application/json, text/plain, */*' \
      -H 'accept-language: en-US,en;q=0.9' \
      -H 'authorization: 1107979040659170353175691805430-1080-1920' \
      -H 'content-type: application/json;charset=UTF-8' \
      -H 'cookie: koa:sess=eyJ1c2VySWQiOjE0NjM2NSwiX2V4cGlyZSI6MTcwNDMzNTgyNzcyOSwiX21heEFnZSI6MjU5MjAwMDAwMDB9; koa:sess.sig=JLqR6VmGfog42y5lZc6hCUIOtjY; ai_user=NyrjYRFYpP1ejPEYLzVAUs|2023-05-23T08:22:56.835Z' \
      -H 'origin: https://glados.rocks' \
      -H 'sec-ch-ua: ".Not/A)Brand";v="99", "Google Chrome";v="103", "Chromium";v="103"' \
      -H 'sec-ch-ua-mobile: ?1' \
      -H 'sec-ch-ua-platform: "Android"' \
      -H 'sec-fetch-dest: empty' \
      -H 'sec-fetch-mode: cors' \
      -H 'sec-fetch-site: same-origin' \
      -H 'user-agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Mobile Safari/537.36' \
      --data-raw '{"token":"glados.network"}' \
      --compressed
