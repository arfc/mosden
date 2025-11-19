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

DATA_DIR="mosden/data/unprocessed"
mkdir -p "$DATA_DIR"

# ENDF --------------------------------------------------------------------
ENDF_VERSION="VII.1"
ALLOWED_VERSIONS=("VIII.0" "VII.1")

if [[ ! " ${ALLOWED_VERSIONS[*]} " =~ " ${ENDF_VERSION} " ]]; then
    echo "Error: Invalid ENDF version '${ENDF_VERSION}'"
    echo "Allowed versions: ${ALLOWED_VERSIONS[*]}"
    exit 1
fi

LOWERCASE_VERSION=$(echo "${ENDF_VERSION//./}" | tr '[:upper:]' '[:lower:]')

ROMAN_PART="${LOWERCASE_VERSION//[0-9]/}"
DIGIT_PART="${LOWERCASE_VERSION//[^0-9]/}"
INTEGER_VALUE=$(roman_to_int "$ROMAN_PART")
LOWERCASE_VERSION="${INTEGER_VALUE}${DIGIT_PART}"

ENDF_DIR="${DATA_DIR}/endfb${LOWERCASE_VERSION}"
NFY_DIR="${ENDF_DIR}"
decay_DIR="${ENDF_DIR}"
mkdir -p "$NFY_DIR"

if [[ "${ENDF_VERSION}" == "VII.1" ]]; then
  SEPARATOR="-"
  decay_SEPARATOR="-"
elif [[ "${ENDF_VERSION}" == "VIII.0" ]]; then
  SEPARATOR="_"
  decay_SEPARATOR="_"
fi

NFY_ZIP_NAME="ENDF-B-${ENDF_VERSION}${SEPARATOR}nfy.zip"
NFY_URL="https://www.nndc.bnl.gov/endf-b${INTEGER_VALUE}.${DIGIT_PART}/zips/${NFY_ZIP_NAME}"

echo "Downloading NFY data for ENDF/B-${ENDF_VERSION}..."
TEMP_ZIP="${NFY_DIR}/${NFY_ZIP_NAME}"
echo "Accessing ${NFY_URL}"
wget --show-progress -O "$TEMP_ZIP" "$NFY_URL"
echo "Extracting NFY data..."
unzip "$TEMP_ZIP" -d "$NFY_DIR"
rm "$TEMP_ZIP"
echo "NFY data handled"

decay_ZIP_NAME="ENDF-B-${ENDF_VERSION}${decay_SEPARATOR}decay.zip"
decay_URL="https://www.nndc.bnl.gov/endf-b${INTEGER_VALUE}.${DIGIT_PART}/zips/${decay_ZIP_NAME}"

echo "Downloading decay data for ENDF/B-${ENDF_VERSION}..."
TEMP_ZIP="${decay_DIR}/${decay_ZIP_NAME}"
echo "Accessing ${decay_URL}"
wget --show-progress -O "$TEMP_ZIP" "$decay_URL"
echo "Extracting decay data..."
unzip "$TEMP_ZIP" -d "$decay_DIR"
rm "$TEMP_ZIP"
echo "Decay data handled"

# /ENDF --------------------------------------------------------------------



# JEFF --------------------------------------------------------------------
JEFF_VERSION="3.1.1"
ALLOWED_VERSIONS=("3.1.1")

if [[ ! " ${ALLOWED_VERSIONS[*]} " =~ " ${JEFF_VERSION} " ]]; then
    echo "Error: Invalid JEFF version '${JEFF_VERSION}'"
    echo "Allowed versions: ${ALLOWED_VERSIONS[*]}"
    exit 1
fi
JEFF_VERSION_NOP="${JEFF_VERSION//./}"

JEFF_DIR="${DATA_DIR}/jeff${JEFF_VERSION_NOP}"
NFY_DIR="${JEFF_DIR}/nfpy/"
mkdir -p "$NFY_DIR"
echo "Saving data to ${NFY_DIR}"

if [[ "${JEFF_VERSION}" == "3.1.1" ]]; then
  JEFF_URL="https://www-nds.iaea.org/public/download-endf/JEFF-${JEFF_VERSION}/nfpy/"
fi


echo "Downloading NFY data for JEFF-${JEFF_VERSION}..."
echo "Accessing ${JEFF_URL}"
wget --show-progress --recursive --no-parent --accept "*.zip" --no-host-directories --cut-dirs=3 -P "${JEFF_DIR}" "$JEFF_URL"
echo "Extracting NFY data..."
for f in "$NFY_DIR"/*.zip; do
    unzip "$f" -d "$NFY_DIR"
done
echo "Removing zip files..."
rm "$NFY_DIR"/*.zip
echo "NFY data handled"


# /JEFF --------------------------------------------------------------------

# IAEA --------------------------------------------------------------------
IAEA_DIR="${DATA_DIR}/iaea"
IAEA_FILE="$IAEA_DIR/eval.csv"
IAEA_URL="https://www-nds.iaea.org/relnsd/delayedn/eval.csv"
mkdir -p "$IAEA_DIR"

echo "Downloading IAEA delayed neutron data..."
wget -q --show-progress -O "$IAEA_FILE" "$IAEA_URL"
echo "Saved to $IAEA_FILE"

# /IAEA --------------------------------------------------------------------

# OpenMC --------------------------------------------------------------------
OPENMC_DIR="${ENDF_DIR}/omcchain/"
mkdir -p "$OPENMC_DIR"
wget -q --show-progress -O "${OPENMC_DIR}chain_casl_pwr.xml" "https://anl.box.com/shared/static/3nvnasacm2b56716oh5hyndxdyauh5gs.xml"
wget -q --show-progress -O "${OPENMC_DIR}chain_casl_sfr.xml" "https://anl.box.com/shared/static/9fqbq87j0tx4m6vfl06pl4ccc0hwamg9.xml"
if [[ "${ENDF_VERSION}" == "VII.1" ]]; then
  wget -q --show-progress -O "${OPENMC_DIR}chain_endfb71_pwr.xml" "https://anl.box.com/shared/static/os1u896bwsbopurpgas72bi6aij2zzdc.xml"
  wget -q --show-progress -O "${OPENMC_DIR}chain_endfb71_sfr.xml" "https://anl.box.com/shared/static/9058zje1gm0ekd93hja542su50pccvj0.xml"
elif [[ "${ENDF_VERSION}" == "VIII.0" ]]; then
  wget -q --show-progress -O "${OPENMC_DIR}chain_endfb71_pwr.xml" "https://anl.box.com/shared/static/nyezmyuofd4eqt6wzd626lqth7wvpprr.xml"
  wget -q --show-progress -O "${OPENMC_DIR}chain_endfb71_sfr.xml" "https://anl.box.com/shared/static/x3kp739hr5upmeqpbwx9zk9ep04fnmtg.xml"
fi
