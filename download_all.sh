#!/bin/bash

mkdir chesapeake_data
cd chesapeake_data

sudo azcopy copy "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/de_1m_2013.zip" ${PWD}/
unzip -q de_1m_2013.zip
rm de_1m_2013.zip

sudo azcopy copy "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/ny_1m_2013.zip" ${PWD}/
unzip -q ny_1m_2013.zip
rm ny_1m_2013.zip

sudo azcopy copy "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/md_1m_2013.zip" ${PWD}/
unzip -q md_1m_2013.zip
rm md_1m_2013.zip

sudo azcopy copy "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/pa_1m_2013.zip" ${PWD}/
unzip -q pa_1m_2013.zip
rm pa_1m_2013.zip

sudo azcopy copy "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/va_1m_2014.zip" ${PWD}/
unzip -q va_1m_2014.zip
rm va_1m_2014.zip

sudo azcopy copy "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/wv_1m_2014.zip" ${PWD}/
unzip -q wv_1m_2014.zip
rm wv_1m_2014.zip

cd ..
