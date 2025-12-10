function I = ChangeBunchCharge(this, laserintensity)

% laser intensity in ratio.
% should be similar to ATF2 panel in %
    
tic

if (laserintensity <= 0.0)
   error('Laser intensity should be strict positive');
end

if (laserintensity >= 1.0)
   error('Laser intensity should be below 1: %d', laserintensity);
end

if (laserintensity >= 0.30)
   error('Laser intensity should be below 0.3 - internal warning %d', laserintensity);
end

ang_offset = 2.0;

angle = 180.0 / pi / 4.0 * acos(2.0*laserintensity - 1.0) - ang_offset;

%%%calculate current angle

%read x_counter

system('caget -# 1 INJ:LaserIntensityXcount > /tmp/bba_laserxcount.txt');

[name,dummy,x_counter] = textread('/tmp/bba_laserxcount.txt', '%s %f %f');

x_counter

angle_read = mod(x_counter,72000) / 200.0 + ang_offset

pulse = int32((angle - angle_read) * 200)

if (pulse >=0)
    disp('laser up');
    system(strjoin(['caput INJ:setLaserIntUpAngle ', num2str(pulse)]));
else
    disp('laser down');
    pulse = -pulse;
    system(strjoin(['caput INJ:setLaserIntDownAngle ', num2str(pulse)]));
end

system('caput INJ:setLaserIntSend 1');
pause(0.5);
system('caput INJ:setLaserIntSend.PROC 1'); 

pause(3)
WriteLog('InterfaceATF2::ChangeBunchCharge()', toc);
I = this;

end
