#!/usr/bin/env bash
# Build the Oazy phone app (an Android APK) from scratch, without Android Studio or Gradle.
#
#   bash phone/build.sh                  # -> phone/dist/oazy.apk
#
# What it needs (Ubuntu 24.04; the first run installs them):
#   apt:  android-sdk-platform-23 aapt dalvik-exchange zipalign apksigner openjdk (javac)
#   pip:  ziglang (a C/C++ cross compiler with its own libc, so no Android NDK is needed)
# The engine is a static ARM64 Linux program (musl libc) that the app starts as a separate process: Android lets an
# app run programs from its own native-library folder, and a static program needs nothing from the phone.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/build"
DIST="$HERE/dist"
SDK=/usr/lib/android-sdk
JAR="$SDK/platforms/android-23/android.jar"

need() { command -v "$1" >/dev/null 2>&1; }
if ! [ -f "$JAR" ] || ! need aapt || ! need dalvik-exchange || ! need zipalign || ! need apksigner; then
  echo "installing the Android build tools (apt) ..."
  sudo_() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi; }
  sudo_ apt-get update -qq
  DEBIAN_FRONTEND=noninteractive sudo_ apt-get install -y -qq android-sdk-platform-23 aapt dalvik-exchange zipalign apksigner default-jdk-headless
fi
python3 -m ziglang version >/dev/null 2>&1 || pip install -q ziglang

rm -rf "$OUT"
mkdir -p "$OUT/lib/arm64-v8a" "$OUT/classes" "$DIST"

echo "[1/5] the engine for ARM64 phones (static, musl) ..."
python3 -m ziglang c++ -target aarch64-linux-musl -O3 -static -std=c++17 -s -fno-exceptions \
  -o "$OUT/lib/arm64-v8a/liboazy.so" "$HERE/engine/oazy.cpp"

echo "[2/5] the app's Java ..."
javac -nowarn -Xlint:-options -source 8 -target 8 -encoding UTF-8 -bootclasspath "$JAR" -d "$OUT/classes" \
  $(find "$HERE/app/src" -name '*.java')

echo "[3/5] to Android's bytecode ..."
dalvik-exchange --dex --min-sdk-version=26 --output="$OUT/classes.dex" "$OUT/classes"

echo "[4/5] packing (manifest, icons, the page, the engine) ..."
aapt package -f -M "$HERE/app/AndroidManifest.xml" -S "$HERE/app/res" -A "$HERE/app/assets" -I "$JAR" \
  -F "$OUT/unsigned.apk" --min-sdk-version 26 --target-sdk-version 34 -0 woff2 -0 png
( cd "$OUT" && aapt add unsigned.apk classes.dex lib/arm64-v8a/liboazy.so >/dev/null )
zipalign -f -p 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"

echo "[5/5] signing ..."
# The app's own signing key. Android only accepts an update signed by the same key, so keep it: it is not in git
# (a private key); the project's copy lives in the project files (phone/signing/oazy.keystore). Point OAZY_KEYSTORE
# at it, or a new key is made here (then uninstall the old app before installing the new build).
KS="${OAZY_KEYSTORE:-$HERE/app/oazy.keystore}"
if [ ! -f "$KS" ] && [ -f /mnt/project-files/phone/signing/oazy.keystore ]; then KS=/mnt/project-files/phone/signing/oazy.keystore; fi
if [ ! -f "$KS" ]; then
  echo "no signing key found: making a new one at $KS"
  keytool -genkeypair -keystore "$KS" -storepass oazyoazy -keypass oazyoazy -alias oazy -keyalg RSA -keysize 2048 \
    -validity 20000 -dname "CN=Oazy AI Research" >/dev/null 2>&1
fi
apksigner sign --ks "$KS" --ks-pass pass:oazyoazy --key-pass pass:oazyoazy --ks-key-alias oazy \
  --out "$DIST/oazy.apk" "$OUT/aligned.apk"
apksigner verify "$DIST/oazy.apk"
echo "built $DIST/oazy.apk ($(du -h "$DIST/oazy.apk" | cut -f1))"
