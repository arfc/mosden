#!/bin/bash
set -euo pipefail

roman_to_int() {
  case "$1" in
    i|I) echo 1 ;;
    ii|II) echo 2 ;;
    iii|III) echo 3 ;;
    iv|IV) echo 4 ;;
    v|V) echo 5 ;;
    vi|VI) echo 6 ;;
    vii|VII) echo 7 ;;
    viii|VIII) echo 8 ;;
    ix|IX) echo 9 ;;
    x|X) echo 10 ;;
    *) echo 0 ;;
  esac
}

download_jeff_data() {
  local JEFF_VERSION="$1"

  local JEFF_VERSION_NOP="${JEFF_VERSION//./}"
  local JEFF_DIR="${DATA_DIR}/jeff${JEFF_VERSION_NOP}"
  local NFY_DIR="${JEFF_DIR}/nfpy/"
  mkdir -p "$NFY_DIR"
  echo "Saving data to ${NFY_DIR}"

  local JEFF_URL
  if [[ "${JEFF_VERSION}" == "3.1.1"  ||  "${JEFF_VERSION}" == "4.0" ]]; then
    JEFF_URL="https://www-nds.iaea.org/public/download-endf/JEFF-${JEFF_VERSION}/nfpy/"
  else
    echo "Unsupported JEFF version: ${JEFF_VERSION}" >&2
    return 1
  fi

  echo "Downloading NFY data for JEFF-${JEFF_VERSION}..."
  echo "Accessing ${JEFF_URL}"
  wget -4 --show-progress --recursive --no-parent --accept "*.zip" --no-host-directories --cut-dirs=3 -P "${JEFF_DIR}" "$JEFF_URL"

  echo "Extracting NFY data..."
  for f in "$NFY_DIR"/*.zip; do
    unzip "$f" -d "$NFY_DIR"


  if [[ "${JEFF_VERSION}" == "4.0" ]]; then
    OPENMC_DIR="${JEFF_DIR}/omcchain/"
    mkdir -p "$OPENMC_DIR"
    echo "Getting OpenMC chain data for JEFF-4.0..."
    wget -4 -q --show-progress -O "${OPENMC_DIR}chain_jeff40_pwr.xml" "https://anl.box.com/shared/static/qpcfyrctoffb34m4dwyxz2vgp8tim8e7.xml"
    wget -4 -q --show-progress -O "${OPENMC_DIR}chain_jeff40_sfr.xml" "https://anl.box.com/shared/static/p6cettxz3ovbp151qg7bc3k9ov0zt3wm.xml"
    echo "OpenMC chain data collected"
  fi
  done

  echo "Removing zip files..."
  rm "$NFY_DIR"/*.zip
  echo "NFY data handled"
}

download_endf_data() {
  local ENDF_VERSION="$1"

  local LOWERCASE_VERSION
  LOWERCASE_VERSION=$(echo "${ENDF_VERSION//./}" | tr '[:upper:]' '[:lower:]')

  local ROMAN_PART="${LOWERCASE_VERSION//[0-9]/}"
  local DIGIT_PART="${LOWERCASE_VERSION//[^0-9]/}"
  local INTEGER_VALUE
  INTEGER_VALUE=$(roman_to_int "$ROMAN_PART")
  LOWERCASE_VERSION="${INTEGER_VALUE}${DIGIT_PART}"

  local ENDF_DIR="${DATA_DIR}/endfb${LOWERCASE_VERSION}"
  local NFY_DIR="${ENDF_DIR}"
  local XS_DIR="${ENDF_DIR}/xs"
  local decay_DIR="${ENDF_DIR}"
  mkdir -p "$NFY_DIR"
  mkdir -p "$XS_DIR"

  local SEPARATOR decay_SEPARATOR XS_URL
  if [[ "${ENDF_VERSION}" == "VII.1" ]]; then
    SEPARATOR="-"
    decay_SEPARATOR="-"
    XS_URL="https://anl.box.com/shared/static/9igk353zpy8fn9ttvtrqgzvw1vtejoz6.xz"
  elif [[ "${ENDF_VERSION}" == "VIII.0" ]]; then
    SEPARATOR="_"
    decay_SEPARATOR="_"
    XS_URL="https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz"
  else
    echo "Unsupported ENDF version: ${ENDF_VERSION}" >&2
    return 1
  fi

  # NFY data
  local NFY_ZIP_NAME="ENDF-B-${ENDF_VERSION}${SEPARATOR}nfy.zip"
  local NFY_URL="https://www.nndc.bnl.gov/endf-b${INTEGER_VALUE}.${DIGIT_PART}/zips/${NFY_ZIP_NAME}"
  local TEMP_ZIP="${NFY_DIR}/${NFY_ZIP_NAME}"
  echo "Downloading NFY data for ENDF/B-${ENDF_VERSION}..."
  echo "Accessing ${NFY_URL}"
  wget -4 --show-progress -O "$TEMP_ZIP" "$NFY_URL"
  echo "Extracting NFY data..."
  unzip "$TEMP_ZIP" -d "$NFY_DIR"
  rm "$TEMP_ZIP"
  if [[ "${ENDF_VERSION}" == "VIII.0" ]]; then
    mv "${NFY_DIR}/ENDF-B-${ENDF_VERSION}${SEPARATOR}nfy" "${NFY_DIR}/nfy"
  fi
  echo "NFY data handled"

  # Decay data
  local decay_ZIP_NAME="ENDF-B-${ENDF_VERSION}${decay_SEPARATOR}decay.zip"
  local decay_URL="https://www.nndc.bnl.gov/endf-b${INTEGER_VALUE}.${DIGIT_PART}/zips/${decay_ZIP_NAME}"
  TEMP_ZIP="${decay_DIR}/${decay_ZIP_NAME}"
  echo "Downloading decay data for ENDF/B-${ENDF_VERSION}..."
  echo "Accessing ${decay_URL}"
  wget -4 --show-progress -O "$TEMP_ZIP" "$decay_URL"
  echo "Extracting decay data..."
  unzip "$TEMP_ZIP" -d "$decay_DIR"
  rm "$TEMP_ZIP"
  if [[ "${ENDF_VERSION}" == "VIII.0" ]]; then
    mv "${decay_DIR}/ENDF-B-${ENDF_VERSION}${decay_SEPARATOR}decay" "${decay_DIR}/decay"
  fi
  echo "Decay data handled"

  # Cross section data
  TEMP_ZIP="${XS_DIR}/XS.tar.xz"
  echo "Downloading cross section data for ENDF/B-${ENDF_VERSION}..."
  wget -4 --show-progress -O "$TEMP_ZIP" "$XS_URL"
  echo "Extracting XS data"
  tar -xvf "$TEMP_ZIP" -C "$XS_DIR" --strip-components=1
  rm "$TEMP_ZIP"
  echo "Cross section data handled"

  # OpenMC chain data
  local OPENMC_DIR="${ENDF_DIR}/omcchain/"
  mkdir -p "$OPENMC_DIR"
  echo "Getting OpenMC chain data for ENDF/B-${ENDF_VERSION}..."
  wget -4 -q --show-progress -O "${OPENMC_DIR}chain_casl_pwr.xml" "https://anl.box.com/shared/static/3nvnasacm2b56716oh5hyndxdyauh5gs.xml"
  wget -4 -q --show-progress -O "${OPENMC_DIR}chain_casl_sfr.xml" "https://anl.box.com/shared/static/9fqbq87j0tx4m6vfl06pl4ccc0hwamg9.xml"
  if [[ "${ENDF_VERSION}" == "VII.1" ]]; then
    wget -4 -q --show-progress -O "${OPENMC_DIR}chain_endfb71_pwr.xml" "https://anl.box.com/shared/static/os1u896bwsbopurpgas72bi6aij2zzdc.xml"
    wget -4 -q --show-progress -O "${OPENMC_DIR}chain_endfb71_sfr.xml" "https://anl.box.com/shared/static/9058zje1gm0ekd93hja542su50pccvj0.xml"
  elif [[ "${ENDF_VERSION}" == "VIII.0" ]]; then
    wget -4 -q --show-progress -O "${OPENMC_DIR}chain_endfb80_pwr.xml" "https://anl.box.com/shared/static/nyezmyuofd4eqt6wzd626lqth7wvpprr.xml"
    wget -4 -q --show-progress -O "${OPENMC_DIR}chain_endfb80_sfr.xml" "https://anl.box.com/shared/static/x3kp739hr5upmeqpbwx9zk9ep04fnmtg.xml"
  fi
  echo "OpenMC chain data collected"
}

DATA_DIR="mosden/data/unprocessed"
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

# ENDF --------------------------------------------------------------------

download_endf_data "VII.1"
download_endf_data "VIII.0"

# /ENDF --------------------------------------------------------------------



# JEFF --------------------------------------------------------------------

download_jeff_data "3.1.1"
download_jeff_data "4.0"

# /JEFF --------------------------------------------------------------------

# IAEA --------------------------------------------------------------------
IAEA_DIR="${DATA_DIR}/iaea"
IAEA_FILE="$IAEA_DIR/eval.csv"
IAEA_URL="https://www-nds.iaea.org/relnsd/delayedn/eval.csv"
mkdir -p "$IAEA_DIR"

echo "Downloading IAEA delayed neutron data..."
wget -4 -q --show-progress -O "$IAEA_FILE" "$IAEA_URL"
echo "Saved to $IAEA_FILE"

# /IAEA --------------------------------------------------------------------

