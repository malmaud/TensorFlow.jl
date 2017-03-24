# Run with
# -v /var/run/docker.sock:/var/run/docker.sock
# -e "prefix=(where you want the images to go")
FROM docker
MAINTAINER Jon Malmaud
ADD cpu /cpu
ADD gpu /gpu
ADD build_libtf.sh /
VOLUME /out
ENV prefix $HOME/out
CMD ["/bin/sh", "/build_libtf.sh"]
