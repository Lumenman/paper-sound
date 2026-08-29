@echo off
rem Цепочка из 01.sh (блок pre-processing audio), перецеленная на 6780 Гц.
rem
rem   prep2.bat [вход.wav] [выход.wav]
rem
rem Отличия от 01.sh только там, где того требует другой носитель:
rem   - без прогона через 22050/pcm_u8: то был формат wav2pbm, а не обработка
rem     звука; paper_sound читает 16 бит, и 8-битный крюк добавил бы только шипение
rem   - rate -v 6780 вместо dpi*21
rem   - trim по ёмкости листа
rem
rem Сама цепочка эффектов — дословно как в 01.sh, включая gain +10,
rem который там клиповал и работал лимитером. Предупреждения sox об этом
rem оставлены намеренно — это часть того звука, а не ошибка запуска.

setlocal
set SOX=%~dp0sox\sox.exe
set IN=%~1
if "%IN%"=="" set IN=input.wav
set OUT=%~2
if "%OUT%"=="" set OUT=out2.wav
set SECS=120

"%SOX%" "%IN%" -b 16 "%OUT%" ^
 trim 0 %SECS% ^
 highpass 100 ^
 compand 0.01,0.15 6:-30,-30,0,-15 4 -90 0.05 ^
 equalizer 2800 1.2q +3 ^
 gain +10 ^
 rate -v 6780
if errorlevel 1 exit /b 1

echo.
"%SOX%" "%OUT%" -n stats
endlocal
