export USERNAME=$1

echo "${USERNAME}"

if [[ $USERNAME == "mieseung" ]]; then
    git config user.name "Miseung Kim"
    git config user.email "mieseung@gmail.com"
fi

if [[ $USERNAME == "footprinthere" ]]; then
    git config user.name "footprinthere"
    git config user.email "tai15515@kakao.com"
fi

