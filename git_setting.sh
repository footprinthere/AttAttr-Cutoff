export USERNAME=$1

echo "${USERNAME}"


if [[ $USERNAME == "footprinthere" ]]; then
    git config --local user.name "footprinthere"
    git config --local user.email "tai15515@kakao.com"
fi
