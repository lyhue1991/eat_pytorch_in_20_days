# -*- coding: utf-8 -*-
# ## 推送主分支

# !git config --global user.email "lyhue1991@163.com"
# 出现一些类似 warning: LF will be replaced by CRLF in <file-name>. 可启用如下设置。
# !git config --global core.autocrlf false
# 配置打印历史commit的快捷命令
# !git config --global alias.lg "log --oneline --graph --all"

# !git init

# !git add -A

# !git pull origin master 

# +
# #!rm -rf *.html
# -

# !git commit -m "revise readme" 

# !git remote add origin git@github.com:lyhue1991/eat_pytorch_in_20_days.git

# !git remote remove origin 

# !git push origin master 

# !ssh -T git@github.com:lyhue1991/eat_pytorch_in_20_days.git

# !git remote rm gitee 

# !git remote add gitee https://gitee.com/Python_Ai_Road/eat_pytorch_in_20_days

# !git push -f  gitee master 

# ## 创建pages分支

# !git checkout -b gh-pages

# !git rm --cached -r *.md

# !git clean -df
# !rm -rf *.md

# !cp -r _book/* .

# !git add .

# !git reset

# !git pull origin gh-pages

# !git commit -m 'add gh-pages'

# !git push -u origin gh-pages

# !git checkout pages







# ## 更新命令

# !git checkout master

# !git add ./data/*  *.md *.py

# !git commit -m "revise readme"

# !git push -u origin master

# !gitbook build

# !git branch -D gh-pages 

# !git checkout -b gh-pages

# !git rm --cached -r *.md

# !git clean -df

# !rm -rf *.md

# !cp -r _book/* .

# !git add .
# !git commit -m "add postscript"

# !git push -f origin gh-pages

# !git checkout master
