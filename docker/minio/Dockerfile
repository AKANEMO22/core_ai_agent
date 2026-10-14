# Local development only: build the last upstream MinIO Community release from source.
# Upstream has archived the Community repository and no longer publishes maintained images.
FROM golang:1.24-alpine AS build

ARG MINIO_RELEASE=RELEASE.2025-10-15T17-29-55Z
ENV CGO_ENABLED=0

RUN apk add --no-cache ca-certificates git
RUN go install -v "github.com/minio/minio@${MINIO_RELEASE}"

FROM alpine:3.22

RUN apk add --no-cache ca-certificates \
    && addgroup -S -g 10001 minio \
    && adduser -S -D -H -u 10001 -G minio minio \
    && mkdir -p /data \
    && chown minio:minio /data

COPY --from=build /go/bin/minio /usr/local/bin/minio

USER 10001:10001
EXPOSE 9000 9001
ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
