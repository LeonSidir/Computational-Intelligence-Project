% mf_X is the vector of values in the membership function input range
% mf_Y is the value of the membership function at mf_X
function defuzz = customdefuzz(mf_X, mf_Y)
	[peaks, locs] = findpeaks(mf_Y, mf_X);
	
	for i = 1:length(peaks)
		% Find the position in the mf_X / mf_Y array where the peak occurs
		StartIndex = find(mf_X == locs(i));
		EndIndex = StartIndex + length(mf_Y(mf_Y == mf_Y(StartIndex))) - 1;
		upper = abs(mf_X(EndIndex) - mf_X(StartIndex));
		area = (0.66 + upper) * peaks(i) / 2; % Surface is trapezoidal 
		COA = (mf_X(EndIndex) + mf_X(StartIndex) ) / 2; % Calculate center of area for the trapezoid
	end
	if (mf_Y(1) ~= 0) % Check NL
		StartIndex = 1;
		EndIndex = StartIndex + length(mf_Y(mf_Y == mf_Y(StartIndex))) - 1; % calculate the number of the times that mf_Y(1) appears in the array
		upper = abs(mf_X(EndIndex) - mf_X(StartIndex));   
		area = (0.33 + upper) * mf_Y(1) / 2; % Surface is trapezoidal 
		f1 = @(w) (mf_Y(1) * w); % constant function
		f2 = @(w) ((mf_Y(1) / (mf_X(EndIndex) + 0.66)) .* (w + 0.66) .* w); 
		Q = integral(f1, - 1, mf_X(EndIndex)) + integral(f2, mf_X(EndIndex), - 0.66);
		COA =  Q / area;
	end
	if (mf_Y(end) ~= 0) % Check PL
			StartIndex = 101;
			EndIndex = StartIndex - length(mf_Y(mf_Y == mf_Y(StartIndex))) - 1;
			upper = abs(- mf_X(EndIndex) + mf_X(StartIndex));
			area = (0.33 + upper) * mf_Y(end) / 2;
			f1 = @(w) ((mf_Y(end) / (mf_X(EndIndex) - 0.66)) .* (w - 0.66) .* w);
			f2 = @(w) (mf_Y(end) * w);
			Q = integral(f1, 0.66, mf_X(EndIndex)) + integral(f2, mf_X(EndIndex), 1);
			COA = Q / area;
	end
	defuzz = COA;
end
