pkg load image

function main2()
    batchsize = 2000000; 
    fs = round( (8000000*8)/7 );
    emulate_delay_samples = 60;
    emulate_doppler_hz = 50;
    max_lag_samples = 70;


    decim = 1; % Downsample factor (reduces data size by X)
    fs_new = fs / decim;

    %test_data_generation();
    apply_doppler_shift( 'new3_no_echo2.iq', 'new3_emulated_doppler.iq', emulate_doppler_hz, emulate_delay_samples, fs )

    file_ref = fopen('new3_no_echo2.iq', 'r');
    file_echo = fopen('new3_emulated_doppler.iq', 'r');

    figure;
    while ~feof(file_ref)
        sig_ref = import_iq( file_ref, batchsize );
        sig_echo = import_iq( file_echo, batchsize );
        if isempty(sig_ref), break; end

        % --- SPEED BOOST: Decimate ---
        % resample() or simple decimation
        sig_ref_d = sig_ref(1:decim:end);
        sig_echo_d = sig_echo(1:decim:end);
        
        % Subtract mean to remove the massive 0-Doppler spike
        sig_ref_d = sig_ref_d - mean(sig_ref_d);
        sig_echo_d = sig_echo_d - mean(sig_echo_d);

        % max_lag_d is max_lag in the new sample rate
        [AF, delays, dopplers] = fast_caf(sig_ref_d, sig_echo_d, fs_new, max_lag_samples); 
        
        % --- ZOOM IN ON DOPPLER ---
        doppler_limit = 100; % Show +/- 100 Hz
        roi_idx = abs(dopplers) <= doppler_limit;
        
        % Crop AF and the dopplers vector
        AF_zoom = AF(roi_idx, :);
        dopplers_zoom = dopplers(roi_idx);

        % Now resize and plot the ZOOOMED version
        new_img = imresize(AF_zoom, [1000 1000]);
        imagesc(delays * 1e6, dopplers_zoom, new_img);
        
        axis xy; colormap(hot); colorbar;
        ylabel('Doppler (Hz)'); xlabel('Delay (\mus)');
        title('Cross-Ambiguity Function (Zoomed)');
        caxis([max(new_img(:))-25, max(new_img(:))]);
        drawnow;
    end
    fclose('all');
end

% Takes as input a gnu radio I/Q file in complex float32 format and add doppler shift and delay
function apply_doppler_shift( fn, fn_new, doppler_hz, delay_samples, fs )
    file_input_ref = fopen( fn, 'rb' ); 
    if file_input_ref == -1
        error( 'Cannot open file: %s', fn );
        return;
    end
    file_output_ref = fopen( fn_new, 'wb' );
    if file_output_ref == -1
        error( 'Cannot open file: %s', fn_new );
        return;
    end

    % Prepend zeros to create delay
    if delay_samples > 0
        initial_zeros = zeros(2, delay_samples, 'single');
        fwrite(file_output_ref, initial_zeros, 'float32');
    end
    
    batchsize = 8192;
    current_sample = 0;
    while ~feof( file_input_ref )
        % Read the data two floats at a time into a 2-row matrix
        data = fread( file_input_ref, [2, batchsize], 'float32');
        if isempty( data )
            break;
        end
        complex_sig = data(1, :) + 1j * data(2, :);
        siglen = length( complex_sig );

        % Generate time vector and apply Doppler Shift
        t = (current_sample + delay_samples : current_sample + delay_samples + siglen - 1) / fs;
        complex_sig = complex_sig .* exp( 1i * 2 * pi * doppler_hz * t );
        
        % Interleave Real/Imaginary and write as float32
        interleaved_data = [ real(complex_sig); imag(complex_sig) ];
        fwrite( file_output_ref, interleaved_data, 'float32' );
        current_sample = current_sample + siglen;
    end
 
   fclose( file_input_ref );
   fclose( file_output_ref );
end


function test_data_generation()
    fs = 2.048e6;
    batchsize = 262144*10;
    
    % 1. Create a Reference Signal (White Noise)
    t = (0:batchsize-1)' / fs;
    sig_ref = (randn(batchsize, 1) + 1i*randn(batchsize, 1));
    
    % 2. Create Echo Signal
    delay_s = 5e-6;
    doppler_hz = 25;
    
    delay_samples = round(delay_s * fs);
    
    % Apply Delay (Padding with noise or zeros)
    sig_echo = [zeros(delay_samples, 1); sig_ref(1:end-delay_samples)];
    
    % Apply Doppler Shift (Frequency shift)
    sig_echo = sig_echo .* exp(1i * 2 * pi * doppler_hz * t);
    
    % Add some noise to the echo
    sig_echo = sig_echo + 0.5 * (randn(batchsize, 1) + 1i*randn(batchsize, 1));

    % 3. Save to files so your main2 can read them
    % Convert to int8 to match your import_iq_int8 function
    write_sigmf_int8('171210ship_ch2.sigmf-data', sig_ref);
    write_sigmf_int8('171210ship_ch1.sigmf-data', sig_echo);
    
    fprintf('Test data created: Delay %d samples, Doppler %d Hz\n', delay_samples, doppler_hz);
end

function write_sigmf_int8(filename, complex_data)
    % Scale data to fit in int8 range (-128 to 127)
    complex_data = complex_data / max(abs(complex_data)) * 120;
    
    % Interleave I and Q
    interleaved = zeros(2 * length(complex_data), 1);
    interleaved(1:2:end) = real(complex_data);
    interleaved(2:2:end) = imag(complex_data);
    
    fid = fopen(filename, 'wb');
    fwrite(fid, interleaved, 'int8');
    fclose(fid);
end

function [AF, delays, dopplers] = fast_caf(sig_ref, sig_echo, fs, max_lag)
    N = length(sig_ref);
    lags = -max_lag:max_lag;
    delays = lags / fs;
    dopplers = (-(N/2):(N/2-1)) * (fs/N);
    
    % 1. Create Hamming window (requires signal package, or use the formula below)
    % formula: 0.54 - 0.46 * cos(2*pi*(0:N-1)'/(N-1))
    win = 0.54 - 0.46 * cos(2 * pi * (0:N-1)' / (N-1));
    
    AF = zeros(N, length(lags));
    r_conj = conj(sig_ref); 
    
    for i = 1:length(lags)
        s_shifted = circshift(sig_echo, -lags(i)); 
        
        % 2. Apply window to the combined signal before FFT
        combined_sig = (s_shifted .* r_conj) .* win;
        
        AF(:, i) = 20*log10(abs(fftshift(fft(combined_sig))) + 1e-6);
    end
end

function [AF, delays, dopplers] = faster_caf(sig_ref, sig_echo, fs, max_lag)
    N = length(sig_ref);
    lags = -max_lag:max_lag;
    delays = lags / fs;
    dopplers = (-(N/2):(N/2-1)) * (fs/N);
    win = 0.54 - 0.46 * cos(2*pi*(0:N-1)'/(N-1));
    
    % FIX: Use subtraction to shift the echo "backwards" for positive lags
    idx = (1:N)' - lags; 
    
    % Circular wrap handling
    idx = mod(idx - 1, N) + 1;
    
    S_matrix = sig_echo(idx); 
    combined = bsxfun(@times, S_matrix, conj(sig_ref) .* win);
    AF = 20*log10(abs(fftshift(fft(combined, N, 1))) + 1e-6);
end

function iq_complex = import_iq( fileID, cnt )
    raw_data = fread(fileID, cnt*2, 'float32');
    if isempty(raw_data), iq_complex = []; return; end
    I = double(raw_data(1:2:end));
    Q = double(raw_data(2:2:end));
    iq_complex = complex(I, Q);
end

main2();
