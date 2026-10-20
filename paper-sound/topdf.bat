@echo off
rem PNG листа -> PDF A4 того же размера, для печати в настоящие 100%.
rem
rem   topdf.bat лист.png [ещё.png ...]   (или перетащить PNG на этот файл)
rem
rem Растр кладётся без пересжатия, 600 dpi записаны в PDF жёстко: 4961x7016 px
rem = ровно A4. Печатать из SumatraPDF/Acrobat с "Фактический размер", не
rem "Подогнать". Нужен img2pdf: pip install img2pdf

setlocal
python -c "import img2pdf" 2>nul || (echo need img2pdf: pip install img2pdf & pause & exit /b 1)
:next
if "%~1"=="" goto done
python -m img2pdf --imgsize 600dpi "%~1" -o "%~dpn1.pdf" || (pause & exit /b 1)
echo %~dpn1.pdf
shift
goto next
:done
endlocal
