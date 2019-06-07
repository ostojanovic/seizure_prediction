
rng default

n = 0:319;
x = cos(pi/4*n)+randn(size(n));

pxx_1 = pwelch(x);
% plot(10*log10(pxx_1))

% specified segment length and segment overlap
segmentLength = 100;
n_overlap = 25;
pxx_2 = pwelch(x, segmentLength, n_overlap);
% figure
% plot(10*log10(pxx_2))

% specified dft length
segmentLength = 100;
nfft = 640;
pxx_3 = pwelch(x, segmentLength, [], nfft);
% figure
% plot(10*log10(pxx_3))
% xlabel('rad/sample')
% ylabel('dB')

% welch psd estimate of signal with frequency in hertz
rng default

fs = 1000;
t = 0:1/fs:5-1/fs;
x = cos(2*pi*100*t)+randn(size(t));

window = 500;    % when WINDOW is a vector, divides each column of X into overlapping sections of the same length as WINDOW, and then 
                 % uses the vector to window each section. 
                 % If WINDOW is an integer, pwelch divides each column of X into sections of length WINDOW, and uses a
                 % Hamming window of the same length. If the length of X is such that it cannot be divided exactly into an integer number 
                 % of sections with 50% overlap, X is truncated. 
                 % A Hamming window is used if WINDOW is omitted or specified as empty.

n_overlap = 300; % number of overlap samples from section to section. N_OVERLAP must be an integer smaller than WINDOW if
                 % WINDOW is an integer, or smaller than the length of WINDOW if WINDOW is a vector. 
                 % If N_OVERLAP is omitted or specified as empty, it is set to obtain a 50% overlap.

n_fft = 500;     % the number of fft points used to calculate the psd estimate

[pxx_4, f] = pwelch(x,window,n_overlap,n_fft,fs);

plot(f,10*log10(pxx_4))
xlabel('Frequency (Hz)')
ylabel('Magnitude (dB)')












