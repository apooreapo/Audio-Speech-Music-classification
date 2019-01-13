function abc = melFilter
    
sampleRate = 22050;
    num = 19;
    nfft = 1024;
    startFreq = 60;
    endFreq = 11025;
    melStartFreq = melTransformation(startFreq);
    melEndFreq = melTransformation(endFreq);
    M = linspace(melStartFreq,melEndFreq,num+2);
    F = inverseMelTransformation(M);
    K(1,:) = floor(1+(nfft+1)*F(1,:)/sampleRate);
    H = zeros(num,513);
    for m = 1:num
        start = K(1,m);
        mid = K(1,m+1);
        stop = K(1, m+2);
        for k = start:mid
            H(m,k) = (k - K(1,m))/(K(1,m+1) - K(1,m));
        end
        for k = mid:stop
            H(m,k) = (K(1,m+2) - k)/(K(1, m+2) - K(1, m+1));
        end
    end
    Ht = transpose(H);
    abc = Ht;


end

function M = melTransformation(f)

    [~,s] =size(f);
    M = zeros(1,s);
    for i =1:s
        M(1,i) = 1125 * log(1 + f/700);
    end
end

function I = inverseMelTransformation(m)

    [~,s] =size(m);
    I = zeros(1,s);
    for i =1:s
        I(1,i) = 700*(exp(m(1,i)/1125)-1);
    end

end