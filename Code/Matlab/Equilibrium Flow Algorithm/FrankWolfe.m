function [MunFlows] = FrankWolfe(Network, Parameters, OriginList, MunLinks, MunIds)
% Parameters ={delta, phi_i, phi_p, phi_t, kappa}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --------------- Step -1: Initialize matrices and prep data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize matrices and tolerance
StoreLowerBound=zeros(2000, 1); %Matrix that will store the value of the interval between the objective at the initial value and the value of the maximized LPP
Conver = 1;
tolerance = .001; %Highly converged
FWiter=0; %start the count for how many times we've looped through FW

% Create sorted list of cost inputs
    % network data: node1 node2 dist bcx port tbcx container commlanes
LinkId=[Network{1}, Network{2}; Network{2}, Network{1}]; %--Link Identifiers
Distance=[Network{3}; Network{3}]; %---Distance
Bcx=[Network{4}; Network{4}]; % Border crossing dummy
Port=[Network{5}; Network{5}]; % Border crossing dummy
Tbcx=[Network{6}; Network{6}]; % Border crossing dummy
Container=[Network{7}; Network{7}]; % Share Container Capacity
Container=Container./10; %Rescale so similar units to lanes
Lanes=[Network{8}; Network{8}]; % Share lanes
Cap = Lanes+Container + (1-Bcx); 
a = Port.*Parameters(3, 1) + Tbcx.*Parameters(2,1); 
Parameters(1,1)=Parameters(1,1)./10;

% Create a single index identifier for each link
id=[1:length(LinkId)]';

% Create a matrix with the inputs for updating the cost function
CostInputs=[LinkId, Distance, a, Cap, id];
clear LinkId Distance Bcx Cap id Container Lanes

% Create a sparse matrix containing a single index identifier for
% each link
LinkIdMatrix=sparse(CostInputs(:, 1), CostInputs(:, 2), CostInputs(:, 6));
        
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --------------- Step 0: Initial paths
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run Dijkstra's algorithm
node1=zeros(20000,1);
node2=zeros(20000,1);
ndoubles=0; 
Distance=sparse(CostInputs(:, 1), CostInputs(:, 2), CostInputs(:, 3)); % Create bi-directonal distance matrice

for j=1:length(OriginList)
    [d pred] = shortest_paths(Distance, OriginList(j), struct('algname','dijkstra'));
    [node1, node2, ndoubles] = PathFlows(OriginList(j), node1, node2, pred, ndoubles); % Convert predecessor matrix to path flows
end

% Aggregate path flows to link flows
[Flows0, Key0] = LinkFlows(node1, node2, ndoubles, LinkIdMatrix);

while (Conver>tolerance)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% ------Step 1: Suproblem to determine search direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Advance the iteration counter
    FWiter=FWiter+1;

    % merge the flows with their relevant cost parameters
    Merged = CostInputs(:, 1:5); % Merged node-node-flow matrix
    Merged(Key0(:,1), 6) = Flows0(:, 1);
    
    % Update costs
    [L1, L2, Cost]=UpdateCosts(Merged, Parameters); %Link costs
    %disp('cost'); mean(Cost)
    CostMatrix=sparse(L1, L2, Cost, 13970, 13970); %Convert to sparse matrix format

    % Run Dijkstra's algorithm
    node1=zeros(20000,1);
    node2=zeros(20000,1);
    ndoubles=0; 
    for j=1:length(OriginList)
        [d pred] = shortest_paths(CostMatrix, OriginList(j), struct('algname','dijkstra'));
        [node1, node2, ndoubles] = PathFlows(OriginList(j), node1, node2, pred, ndoubles); % Convert predecessor matrix to path flows
    end

    clear CostMatrix Distance 
    
    % Aggregate path flows to link flows 
    [Flows1, Key1] = LinkFlows(node1, node2, ndoubles, LinkIdMatrix);
    
    % Merge link flows with cost parameters and keep only positive flows
    Merged(Key1(:, 1), 7) = Flows1(:, 1);
    Merged(:, 8) = CostInputs(:, 6); 
    [D, B, C, F0, F1, Key0, Cost]=ParseBothFlows(Merged, Cost);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% ------Step 2: Step-size determination (line search)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    % Combine flows
    rho = fminbnd(@(rho) objective(rho,D, B, C, F0, F1, Parameters), 0, 1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%-----Step 3: Check for convergence
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Gap = sum((F1-F0) .*Cost); 
    StoreLowerBound(FWiter, 1) = objective(0, D, B, C, F0, F1, Parameters) + Gap; 
    Conver = - Gap./abs(max(StoreLowerBound(1:FWiter, 1)))

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%-----Step 4: Move - x(n+1)=x(n)+rho(n)(y(n)-x(n))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Flows0=F0+rho*(F1-F0); 

end

%%%%%%%%%%%%%%%%%%%%%%%
%%%---Step 5: Create Mun flows
%%%%%%%%%%%%%%%%%%%%%%%

% Run Dijkstra's with the final iterate, saving path flows by mun rather than link flows

% merge the flows with their relevant cost parameters
Merged = CostInputs(:, 1:5); % Merged node-node-flow matrix
Merged(Key0(:,1), 6) = Flows0(:, 1);

% Update costs
[L1, L2, Cost]=UpdateCosts(Merged, Parameters); %Link costs

CostMatrix=sparse(L1, L2, Cost, 13970, 13970); %Convert to sparse matrix format

% Prepare mun inputs
EdgeMunId=MunInputs(MunLinks, LinkIdMatrix);

% Run Dijkstra's algorithm
[dEM1, dEM2]=size(EdgeMunId);
PathMun=zeros(length(OriginList), dEM2);
for j=1:length(OriginList)
    [d pred] = shortest_paths(CostMatrix, OriginList(j), struct('algname','dijkstra'));
    PathMun = Path2Mun(OriginList(j), j, pred,LinkIdMatrix, EdgeMunId, PathMun, dEM1);
end

% Aggregate mun flows across paths
MunFlows=sum(PathMun);
MunFlows=MunFlows'; 
MunFlows=[MunIds{1},  MunFlows]; 

%%%%%%%%%%%%%%%%%%%
%%%------------------------Subfunctions
%%%%%%%%%%%%%%%%%%%

    function [node1, node2, ndoubles] = PathFlows(OriginList, node1, node2, pred, ndoubles)

        i = 13970; 
         if pred(i)~=0
           while i ~= OriginList
                ndoubles=ndoubles+1; 
                len=length(node1);
                if (ndoubles>len)
                    node1(2*len)=0; 
                    node2(2*len)=0;
                end
                node1(ndoubles, 1) = pred(i);
                node2(ndoubles, 1) = i; 
                i=pred(i); 
           end 
         end
    end    

    function [EdgeMunId]=MunInputs(MunLinks, LinkIdMatrix)
  
        % Make mun-edge pairs bi-directional
        %InegiKey=[MunLinks{1}; MunLinks{1}]; % st_mun_id
        MunId=[MunLinks{2}; MunLinks{2}]; MunNode=[MunLinks{3}, MunLinks{4}; MunLinks{4}, MunLinks{3}]; %bi-directional node and mun ids
        
        % Merge Mun-edge pairs (currently in mun-mun-node form) with the single edge id
        [d1 d2]=size(LinkIdMatrix); % for creating single index for LinkIdMatrix;
        SingleIndex = (MunNode(:, 2)- 1).*d1+ MunNode(:, 1); 
        EdgeId=LinkIdMatrix(SingleIndex); 
        EdgeId=full(EdgeId);  MunId=MunId(EdgeId>0); EdgeId=EdgeId(EdgeId>0); 
        clear d1 d2 SingleIndex MunNode MunLinks;
        EdgeMunId=sparse(EdgeId, MunId, ones(length(MunId), 1)); 

    end


    function [PathMun] = Path2Mun(OriginList, j, pred,LinkIdMatrix, EdgeMunId, PathMun, dEM1)
        
        % Initialize inputs
        LinkId_j=zeros(500,1);
        nlid=0; 
        i = 13970; 
        
        % Make a list of edges in the optimal path from OriginList(j) - converts lists of nodes traversed into
        % a list of edges traversed
        if pred(i)~=0
            while i ~= OriginList
                nlid=nlid+1; 
                len=length(LinkId_j);
                if (nlid>len)
                    LinkId_j(2*len)=0; 
                end
                LinkId_j(nlid, 1) = LinkIdMatrix(pred(i), i);
                i=pred(i); 
            end 
        LinkId_j=LinkId_j(1:nlid, 1); % list of edges in each path
     
           % Convert list of edges in the path into a list of Muns in the path
        [dPM1, dPM2]=size(PathMun); % This is the matrix that we need to fill in
        M=[1:dPM2]'; % Column index for PathMun; max(M)==dMP2
        %pxm_Index=(M-ones(dPM2, 1)).*dPM1+ones(dPM2, 1).*j;  % Linear index for filling in jth row of Path Muns
  
        PathMunJ=zeros(dPM2, 1); % Fill in a single row of PathMun so can use PathMun(PathMun>1)=1
        for Lj=1:length(LinkId_j) 
            exm_Index=(M-ones(dPM2, 1)).*dEM1+LinkId_j(Lj).*ones(dPM2, 1);
            PathMunJ = EdgeMunId(exm_Index)+PathMunJ;
        end
        PathMunJ(PathMunJ>1)=1; % Don't double count paths with multiple edges in same mun
        PathMun(j, :) = PathMunJ'; 
        end
    end    

    function [Flows, Key] = LinkFlows(node1, node2, ndoubles, LinkIdMatrix)
    
        % Aggregate path flows to link flows using a sparse
        % matrix
        LinkFlowMatrix=sparse(node1(1:ndoubles), node2(1:ndoubles), ones(ndoubles, 1));

        % Convert them to list format with unique identifier
        [unode1, unode2, Flows]=find(LinkFlowMatrix);
    
        Key = zeros(length(Flows), 1);
        for i=1:length(Flows)
            Key(i) = LinkIdMatrix(unode1(i), unode2(i)); 
        end
    
    end

    function[L1, L2, Cost]=UpdateCosts(Merged, Parameters)
        
        % Want to select on positive flows
        F=Merged(:, 6); % Flows
        L1=Merged(:, 1); % Link id 1
        L2=Merged(:,2);  % Link id 2
        D0=Merged(:, 3);  % Distance
        alpha=Merged(:, 4); % conversion param (P2, P3); = 0 for non-bcx
        C0=Merged(:,5);  % Capacity
        P1=Parameters(1,1);
        
        % Update costs using the flows from the current iterate
        Cost =  D0 + alpha.*(F ./C0).^P1;
    end

    function[D, B, C, F0, F1, ID, Cost]=ParseBothFlows(Merged, Cost)
        % Want to select on positive flows
        positive=Merged(:, 6) + Merged(:, 7); % selector for positive flows
        F0=Merged(:, 6); % Flows initial
        F1=Merged(:, 7); % Flows linear objective
        D=Merged(:, 3);  % Distance
        B=Merged(:, 4);  % alpha
        C=Merged(:,5);  % cap
        ID=Merged(:, 8); % ID

        D=D(positive>0);
        B=B(positive>0);
        C=C(positive>0);
        F0=F0(positive>0); 
        F1=F1(positive>0);
        ID=ID(positive>0);
        Cost=Cost(positive>0); 
    end

    function ValueObjective = objective(rho,D, B, C, F0, F1, Parameters)
                 
        % Make parameters easier to read
        P1=Parameters(1,1);
        z=F0+rho.*(F1-F0); 
        
        % Plug into the closed form solution for f=Bcx .*Parameters(2,1).*(p .*Cap).^Parameters(4,1) + Nonbcx .*Dist .* (1+Parameters(1,1).*(p .*Cap).^Parameters(3,1));
        ValueObjective =sum(D.*z + (B.*z.*(z./C).^P1)./(P1 + 1));

    end

end

