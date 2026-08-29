@echo off
rem Середина между prep.bat и prep2.bat: плотность ближе к prep, характер -
rem к prep2. Ухо предпочло prep2, но тот отдаёт ~2.6 dB среднего уровня на
rem бумаге (крест 7.5 против 4.9); виновником звука prep оказался подъём
rem -40 -> -14 (+26 dB) в первом compand, а не финальный клип.
rem
rem   prep3.bat [вход.wav] [выход.wav]
rem
rem Отличие от prep.bat одно: подъём тихого смягчён - -40 -> -22 (+18 dB) и
rem -20 -> -12 (+8) вместо -40 -> -14 и -20 -> -9. Гейт пауз (-65 -> -90)
rem оставлен как был. Плотность добирается клипом, см. меню ниже.

setlocal
set SOX=%~dp0sox\sox.exe
set IN=%~1
if "%IN%"=="" set IN=input.wav
set OUT=%~2
if "%OUT%"=="" set OUT=out3.wav
set SECS=120

rem СКОЛЬКО ЖАТЬ. Измерено на input.wav со смягчённым compand выше
rem (для сравнения: prep.bat даёт RMS -14.7 crest 4.9, prep2 - RMS -17.5
rem crest 7.5):
rem
rem   6:-12,-10,0,-6   RMS -15.5  crest 5.3   текущая: +2.2 dB к prep2 на бумаге
rem   6:-9,-8,0,-5     RMS -17.0  crest 6.3   +1.2 dB к prep2
rem   6:-6,-5,0,-3     RMS -18.4  crest 7.4   плотность prep2 без его клипа
rem
rem Тише = чище на слух, но шум бумаги громче относительно речи.
set CLIP=6:-12,-10,0,-6

"%SOX%" "%IN%" -b 16 "%OUT%" ^
 trim 0 %SECS% ^
 gain -6 ^
 equalizer 2600 1.0q +4 ^
 highpass 80 ^
 compand 0.002,0.25 6:-65,-90,-40,-22,-20,-12,0,-5 0 -90 0.1 ^
 compand 0,0 %CLIP% 0 -90 ^
 rate -v 6780 ^
 gain -n -1
if errorlevel 1 exit /b 1

echo.
"%SOX%" "%OUT%" -n stats
endlocal
