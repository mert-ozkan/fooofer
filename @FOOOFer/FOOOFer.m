classdef FOOOFer < handle

    % O1OFFitter Class for parameterizing neural power spectra.
    % Implements a FOOOF-like algorithm with multi-pass peak detection.
    % Peak amplitudes are fitted in log-space internally but stored and outputted in linear space.

    properties

        % Peak Fit Settings
        log_scale (1,1) logical = true
        error_distribution {mustBeText, mustBeMember(error_distribution, {'gaussian', 'normal', 'gamma'})} = 'gaussian'
        max_refit_n_iter = 10
        min_peak_width (1,1) double = 1
        max_n_peaks (1,1) double = 5              % Maximum number of peaks to fit per pass
        min_peak_distance (1,1) double = 1.0 % Minimum Hz separation to keep distinct peaks
        min_peak_frequency (1,1) double = NaN
        max_peak_frequency (1,1) double = NaN
        excluded_frequencies (:,2) double = []

        results = struct( ...
            iter = [], type = [], seed = [], ...
            fit = [], fit_flag = [], ...
            fit_n_iter = [], fit_n_func_eval = [], fit_n_pcg_iter = [],...
            gof = [], gof_metric = [], modelSurvived = [])

        verbose = true;

        % fmincon options
        max_func_eval = 50
        max_fit_iter = 50;

    end

    properties (SetAccess = protected)

        iter = 0
        periodic_model SumOfGaussians
        aperiodic_model ExponentialPowerLaw
        periodic_minimizer
        aperiodic_minimizer
        
    end

    properties (Dependent)

        nll_minimizer function_handle
        includeBaseline
        includeKnee

    end

    methods

        [results, ap_fitter, p_fitter] = fit(obj, freqs, spectrum, include_freq_range, exclude_freq_range, apriori_peak_range)

    end

    methods

        function obj = FOOOFer(varargin)
            % Constructor for FOOOFer

            if nargin > 0
                if mod(nargin, 2) ~= 0
                    error('FOOOFer:InvalidInput', 'Invalid number of input arguments. Must be name-value pairs.');
                end
                for i = 1:2:nargin
                    prop_name = varargin{i};
                    prop_val = varargin{i+1};
                    if isprop(obj, prop_name)
                        obj.(prop_name) = prop_val;
                    else
                        warning('FOOOFer:InvalidProperty', 'Property "%s" does not exist.', prop_name);
                    end
                end
            end

            % initiate inner models and the minimizer
            obj.initiate()

        end

        function initiate(obj, pv)

            arguments
                obj
                pv.includeKnee = false
                pv.n_peaks = obj.max_n_peaks

            end

            % initiate inner models and the minimizer

            if obj.log_scale

                scale = 'log';

            else

                scale = 'linear';

            end

            % no baseline parameter when the model is multiplicative
            includeBaseline = ~strcmp(obj.error_distribution, 'gamma');

            obj.aperiodic_model = ExponentialPowerLaw(includeKnee=pv.includeKnee, ...
                scale=scale, verbose=false);
            obj.periodic_model = SumOfGaussians(n_peaks = pv.n_peaks, ...
                min_peak_width = obj.min_peak_width,...
                min_peak_distance = obj.min_peak_distance,...
                min_peak_frequency = obj.min_peak_frequency,...
                max_peak_frequency = obj.max_peak_frequency,...
                includeBaseline=includeBaseline,...
                verbose = false);

            obj.aperiodic_minimizer = obj.nll_minimizer(obj.aperiodic_model);
            obj.periodic_minimizer = obj.nll_minimizer(obj.periodic_model);

        end

        function next(obj)

            if obj.iter >= obj.max_refit_n_iter
                return;

            end
            obj.iter = obj.iter + 1;
            % update tables

        end

        function append_to_results(obj, pv)

            arguments

                obj

                pv.iter = []
                pv.type = []
                pv.seed = []
                pv.fit  = []
                pv.fit_flag = []
                pv.fit_n_iter = []
                pv.fit_n_func_eval = []
                pv.fit_n_pcg_iter = []
                pv.gof = []
                pv.gof_metric = []
                pv.modelSurvived = []

                pv.nextEntry = false
                pv.step_from_current_entry = 0;

            end

            fld_names = fieldnames(pv);

            n_fld = numel(fld_names);
            allowed_fieldnames = fieldnames(obj.results);
            n_rows = numel(obj.results);

            % if nextEntry=true, append the next entry unless it is the first entry
            if pv.nextEntry & ~isempty(obj.results(1).iter)
                iRow = n_rows + 1;
            else
                iRow = n_rows + pv.step_from_current_entry;
            end

            for ii = 1:n_fld

                fldN = fld_names{ii};
                valN = pv.(fldN);
                if ~ismember(fldN, allowed_fieldnames) || isempty(valN) || all(ismissing(valN))
                    continue;
                end

                obj.results(iRow).(fldN) = valN;

            end



        end

        function varargout = simulate(self, x, ap_p, p_p, err_p, noise_dist)

            arguments
                
                self
                x (:,:) double
                ap_p (1,:) double
                p_p (:,:) double
                err_p (1,1) double % error param, either k or sigma
                noise_dist {mustBeMember(noise_dist,{'normal', 'gaussian', 'gamma'})} = self.error_distribution

            end

            ap = self.aperiodic_model.predict(ap_p, x);

            p_mdl_predict_args = {p_p(1:end-self.includeBaseline)};
            if self.includeBaseline
                p_mdl_predict_args{2} = p_p(end);
            end
            p = self.periodic_model.predict(p_mdl_predict_args{:}, x);
            
            if strcmp(noise_dist,'gamma')

                y_clean = ap.*p;
                y_sim =  gamrnd(err_p, y_clean./ err_p);

            else

                y_clean = ap + p;
                y_sim = 10.^(y_clean + normrnd(0, err_p, size(x)));
                y_clean = 10.^y_clean;

            end

            varargout = cell(1,nargout);
            varargout{1} = y_sim;
            if nargout > 1
                varargout{2} = y_clean;
            end


        end
        

        % Get/Set Methods
        function m = get.nll_minimizer(self)

            % NLLMinimizer constructor function

            if strcmp(self.error_distribution, 'gaussian')

                m = @GaussianNLLMinimizer;

            elseif ~self.log_scale && strcmp(self.error_distribution, 'gamma')

                m = @GammaNLLMinimizer;

            else

                error('Error distribution of the NLLMinimizer (i.e., gamma or gaussian) must match the data scale (linear versus logarithmic)!')
            
            end

        end

        function set.log_scale(self, val)

            arguments
               
                self
                val (1,1) logical

            end

            if self.log_scale ~= val

                % change inner model 

            end
            self.log_scale = val;
            
        end

        function i = get.includeBaseline(self)

            i = self.periodic_model.includeBaseline;

        end

        function i = get.includeKnee(self)

            i = self.aperiodic_model.includeKnee;
            
        end


    end

    methods (Access = protected)

        function p = isolate_component_(self, full_spectrum, comp_spectrum)

            % ISOLATE_PERIODIC_ decides whether to subtract (log scaled data)
            % or divide (linearly scaled data) the aperiodic or periodic 
            % component from the full spectra
            if self.log_scale

                p = full_spectrum - comp_spectrum;

            else

                p = full_spectrum ./ comp_spectrum;

            end
           
        end

        function p = combine_components_(self, full_spectrum, comp_spectrum)

            if self.log_scale

                p = full_spectrum + comp_spectrum;

            else

                p = full_spectrum .* comp_spectrum;

            end

        end

    end



    methods (Static)
        function aic = calculate_aic_(residuals, n_param)

            residuals = residuals(:);
            n_sample = numel(residuals);
            rss = sum(residuals.^2);
            log_lik = -n_sample / 2 * (log(2*pi)+log(rss/n_sample) + 1);
            aic = 2*n_param - 2*log_lik;
            if n_sample / n_param < 40
                % correction for low sample sizes
                aic = aic + (2*n_param*(n_param + 1)) / (n_sample - n_param - 1);

            end
        end

        function bic = calculate_bic_(residuals, n_param)

            residuals = residuals(:);
            n_sample = numel(residuals);
            bic = log(n_sample) * n_param + n_sample .* log(sum(residuals.^2) ./ n_sample);

        end

        function r2 = calculate_r2(y, y_hat, n_param)

            n_sample = numel(y);
            residuals = y - y_hat;
            residuals = residuals(:);
            ss_res = sum(residuals.^2);
            ss_total = sum((y(:) - mean(y(:))).^2);
            if ss_total < 1e-12; r2 = 1; else; r2 = 1 - (ss_res / ss_total); end

            % adjust according to no of params
            r2 = 1 - (((1 - r2) * (n_sample - 1)) / (n_sample - n_param - 1));

        end



        function [R2, AIC, BIC] = calculate_gof(y, y_hat, n_param)

            residuals = y - y_hat;
            R2 = FOOOFer.calculate_r2(y, y_hat, n_param);
            AIC = FOOOFer.calculate_aic_(residuals, n_param);
            BIC = FOOOFer.calculate_bic_(residuals, n_param);

        end

        


    end

end