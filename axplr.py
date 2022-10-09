import sys
import grasptext as grasp
import numpy as np
from typing import Iterable, List, Set, Callable, Optional, Union, Sequence, Dict, Tuple
from toposort import toposort_flatten
from sklearn.linear_model import LinearRegression, LogisticRegression 
import plotly.offline as py 
import plotly.graph_objs as go 
import random

class Argument():
    
    @classmethod
    def set_base_score_function(cls, func: Callable) -> None:
        cls.compute_base_score = func
    
    @staticmethod
    def compute_base_score(p: grasp.Pattern, feedback: List[int] = [], is_default: bool = False) -> float:
        raise NotImplementedError("Please set the base score function before creating an argument.")
    
    def __init__(self, 
                 idx: int, 
                 p: grasp.Pattern, 
                 feedback: List[int] = [], 
                 base_score: float = None,
                 support_class: str = None,
                 is_default: bool = False) -> None:
        self.id = idx
        self.pattern = p
        self.feedback = feedback
        self.is_default = is_default
        self.support_class = support_class if support_class is not None else p.support_class 
        if base_score is not None:
            self.base_score = base_score 
        else:
            try:
                self.base_score = Argument.compute_base_score(p, feedback, is_default)
            except NotImplementedError:
                self.base_score = None
    
    def add_new_feedback(self, new_feedback: List[bool]) -> None:
        old_score = self.base_score
        self.feedback.extend([bool(f) for f in new_feedback])
        self.base_score = Argument.compute_base_score(self.pattern, self.feedback)
        print(f"The base score is updated from {old_score} to {self.base_score}")
        

class QBAF():
    
    def __init__(self,
                 args: List[Argument],
                 atts: Set[Tuple[int, int]],
                 sups: Set[Tuple[int, int]],
                 semantics: str = 'dfquad' # 'dfquad' or 'euler' or 'logistic' or 'logistic-x'
                ) -> None:
        
        self.args = args
        self.semantics = semantics
        self.atts, self.sups = atts, sups
        self.rels = atts.union(sups)
    
    @staticmethod
    def dfquad_strength_aggregation(strs: List[float]) -> float:
        if strs == []: 
            return 0
        elif len(strs) == 1: 
            return strs[0]
        elif len(strs) == 2: 
            return strs[0] + strs[1] - (strs[0] * strs[1])
        else: 
            first = QBAF.dfquad_strength_aggregation(strs[:-1])
            return QBAF.dfquad_strength_aggregation([first, strs[-1]])
    
    @staticmethod
    def dfquad(base: float,
               att_strs: List[float],
               sup_strs: List[float]
              ) -> float:
        all_att = QBAF.dfquad_strength_aggregation(att_strs)
        all_sup = QBAF.dfquad_strength_aggregation(sup_strs)
        if all_att >= all_sup:
            return base * (1.0 - np.abs(all_sup-all_att))
        else:
            return base + ((1.0 - base) * np.abs(all_sup-all_att))
    
    @staticmethod
    def euler(base: float,
               att_strs: List[float],
               sup_strs: List[float]
              ) -> float:
        energy = sum(sup_strs) - sum(att_strs)
        return 1.0 - ((1.0 - base**2) / (1.0 + base * np.exp(energy)))
    
    @staticmethod
    def logistic(base: float,
               att_strs: List[float],
               sup_strs: List[float]
              ) -> float:
        return base + sum(sup_strs) - sum(att_strs)
    
    def compute_strengths(self):
        # Find order to compute the final strength using topological sort
        self.rely_on = {}
        for e in self.rels:
            if e[1] in self.rely_on:
                self.rely_on[e[1]].add(e[0])
            else:
                self.rely_on[e[1]] = {e[0]}
        order = toposort_flatten(self.rely_on)
        if order == []: # There is no other argument except the default
            assert len(self.args) == 1
            order = [0]
        print(f"Compute order: {order}")
        
        # Check attacks and supports of each argument
        self.attacked, self.supported = {aidx: set() for aidx in self.args}, {aidx: set() for aidx in self.args}
        for e in self.atts:
            self.attacked[e[1]].add(e[0])
        for e in self.sups:
            self.supported[e[1]].add(e[0])
        print(f"Attacked: {self.attacked}")
        print(f"Supported: {self.supported}")
        
        # Check out degrees
        self.out_degrees = {aidx: 0 for aidx in self.args}
        for e in self.rels:
            self.out_degrees[e[0]] += 1
        
        # Compute the final strength of each argument based on the topological sort order
        self.final_strengths = {aidx: None for aidx in self.args}
        for aidx in order:
            SEMANTICS = {'dfquad': QBAF.dfquad, 'euler': QBAF.euler, 'logistic': QBAF.logistic, 'logistic-x': QBAF.logistic, 'logistic-1': QBAF.logistic, 'logistic-2': QBAF.logistic}
            if self.semantics in ['logistic', 'logistic-1']:
                att_strs = [self.final_strengths[bidx]/self.out_degrees[bidx] for bidx in self.attacked[aidx]]
                sup_strs = [self.final_strengths[bidx]/self.out_degrees[bidx] for bidx in self.supported[aidx]]
            elif self.semantics in ['logistic-x', 'logistic-2']:
                att_strs = [self.final_strengths[bidx]/max(self.out_degrees[aidx],1) for bidx in self.attacked[aidx]]
                sup_strs = [self.final_strengths[bidx]/max(self.out_degrees[aidx],1) for bidx in self.supported[aidx]]
            else:
                att_strs = [self.final_strengths[bidx] for bidx in self.attacked[aidx]]
                sup_strs = [self.final_strengths[bidx] for bidx in self.supported[aidx]]
            self.final_strengths[aidx] = SEMANTICS[self.semantics](self.args[aidx].base_score, att_strs, sup_strs)

        # Check that all args have been computed
        assert all([f is not None for f in self.final_strengths.values()]), "Some arguments have not been computed the final strengths" + str(self.final_strengths)
        print(f"Final strengths: {self.final_strengths}")
        
    def predict_proba(self):
        self.compute_strengths()
        if 'logistic' in self.semantics: # Apply sigmoid function
            final_strength_default = 1 / (1 + np.exp(-self.final_strengths[0])) 
        else:
            final_strength_default = self.final_strengths[0]
        assert final_strength_default >= 0 and final_strength_default <= 1
        if self.args[0].support_class == 'Negative':
            return [final_strength_default, 1-final_strength_default]
        else:
            return [1-final_strength_default, final_strength_default]   
        
        
class ArgTextClas():
    
    def __init__(self,
                 semantics: str = 'dfquad',
                 pattern_to_base_score_func: Callable = None,
                 reverse: bool = False 
                ) -> None:
        
        self.semantics = semantics
        self.all_args = None
        self.reverse = reverse
        if pattern_to_base_score_func is not None:
            Argument.set_base_score_function(pattern_to_base_score_func)
    
    @staticmethod
    def from_grasp(grasp_model: grasp.GrASP,
                   the_patterns: List[grasp.Pattern],
                   semantics: str = 'dfquad',
                   X_train: List[str] = None, 
                   y_train: List[int] = None,
                   pattern_to_base_score_func: Callable = None,
                   reverse: bool = False 
                  ) -> 'ArgTextClas':
        classifier = ArgTextClas(semantics, pattern_to_base_score_func, reverse)
        classifier.grasp_model = grasp_model
        classifier.the_patterns = the_patterns
        classifier.X_train = X_train
        classifier.y_train = y_train
        classifier._setup()
        return classifier
        
    @staticmethod
    def from_data(grasp_config: Dict[str, object] = dict(),
                   semantics: str = 'dfquad',
                   X_train: List[str] = None, 
                   y_train: List[int] = None,
                   pattern_to_base_score_func: Callable = None,
                   reverse: bool = False 
                  ) -> 'ArgTextClas':
        classifier = ArgTextClas(semantics, pattern_to_base_score_func, reverse)
        classifier.grasp_config = grasp_config
        classifier.X_train = X_train
        classifier.y_train = y_train
        
        # Run GrASP
        classifier.positive = [t for idx, t in enumerate(X_train) if y_train[idx]]
        classifier.negative = [t for idx, t in enumerate(X_train) if not y_train[idx]]
        classifier.grasp_model = grasp.GrASP(**classifier.grasp_config)
        classifier.the_patterns = classifier.grasp_model.fit_transform(classifier.positive, classifier.negative)
        classifier._setup()
        return classifier
        
    def _setup(self) -> None:
        # 0. Fit logistic regressions for logistics semantics (including logistic-x)
        if 'logistic' in self.semantics:
            assert self.X_train is not None
            assert self.y_train is not None
            X_train_f = grasp.extract_features(self.X_train,
                     patterns = self.the_patterns,
                     polarity = False, 
                     include_standard = self.grasp_model.include_standard, 
                     include_custom = self.grasp_model.include_custom)
            self.reg = LogisticRegression().fit(X_train_f, self.y_train)
            
        # 1. Create arguments + Default argument
        self.all_args = [Argument(0, self.grasp_model.root_pattern, is_default = True)]
        for pidx, p in enumerate(self.the_patterns):
            self.all_args.append(Argument(pidx+1, p, is_default = False))
        if 'logistic' in self.semantics:
            for aidx, alpha in enumerate(self.all_args):
                if aidx == 0:
                    alpha.base_score = np.abs(self.reg.intercept_[0])
                    alpha.support_class = "Positive" if self.reg.intercept_[0] >= 0 else "Negative"
                else:
                    alpha.base_score = np.abs(self.reg.coef_[0,aidx-1])
                    alpha.support_class = "Positive" if self.reg.coef_[0,aidx-1] >= 0 else "Negative"
        
        # 2. Draw specialization relations
        self.more_specific_edges = set()
        for alpha in self.all_args:
            for beta in self.all_args:
                if alpha.id == beta.id:
                    continue
                if grasp.is_specialized(alpha.pattern, beta.pattern):
                    self.more_specific_edges.add((alpha.id, beta.id))
        
        # 3. Remove indirect specialization relations
        to_remove = set()
        E = self.more_specific_edges
        for edge in E:
            for i in range(len(self.all_args)):
                if (edge[0], i) in E and (edge[1], i) in E:
                    to_remove.add((edge[0], i))
        self.more_specific_edges.difference_update(to_remove)

        # 4. Normalize base scores by out-degrees for logistic-x
        if self.semantics == 'logistic-x':
            self.all_out_degrees = {a.id: 0 for a in self.all_args}
            for e in self.more_specific_edges:
                self.all_out_degrees[e[0]] += 1
            for aidx, alpha in enumerate(self.all_args):
                if aidx != 0:
                    alpha.base_score /= self.all_out_degrees[aidx]
        
    def predict_proba(self, texts: List[str]):
        # 1. Find matched patterns
        matches = grasp.extract_features(texts,
                     patterns = [a.pattern for a in self.all_args],
                     polarity = False, 
                     include_standard = self.grasp_model.include_standard, 
                     include_custom = self.grasp_model.include_custom)
        
        # 2. Predict for each input text
        results = []
        self.recent_test_qbafs = []
        for eidx, t in enumerate(texts):
            print(f"Eidx: {eidx}")
            the_args = {aidx:a for aidx, a in enumerate(self.all_args) if matches[eidx,aidx]}
            the_atts, the_sups = set(), set()
            if self.reverse: # Phrases to the top
                relevant_edges = [(e[1], e[0]) for e in self.more_specific_edges if (e[0] in the_args and e[1] in the_args and e[1] != 0)]
            else: # Words to the top
                relevant_edges = [e for e in self.more_specific_edges if (e[0] in the_args and e[1] in the_args)]
            
            # Ensure that all arguments (except the root) has at least one out-going edge
            # This aims to solve the imperfect is_specialized which is based on training data
            has_outgoing_edge = set([e[0] for e in relevant_edges])
            for aidx in the_args:
                if aidx != 0 and aidx not in has_outgoing_edge:
                    relevant_edges.append((aidx, 0))

            # Divide all the relevant edges to attackss and supports        
            for e in relevant_edges:    
                if the_args[e[0]].support_class != the_args[e[1]].support_class:
                    the_atts.add(e)
                else:
                    the_sups.add(e)
                    
            the_qbaf = QBAF(the_args, the_atts, the_sups, semantics = self.semantics)
            results.append(the_qbaf.predict_proba())
            self.recent_test_qbafs.append(the_qbaf)
            print('-'*50)
        return np.array(results)
    
    def predict(self, texts: List[str]):
        return self.predict_proba(texts).argmax(axis=1).squeeze()

# From QBAF to local explanation
def postprocess(q: QBAF, 
                flipNeg: bool = True,
                divideStr: bool = False,
                ) -> QBAF:
    # Step 0: Copy the basic attributes from the original QBAF
    assert q.semantics in ['logistic', 'logistic-x']
    the_args = {aidx:Argument(a.id, a.pattern, feedback = a.feedback, base_score = a.base_score, support_class = a.support_class, is_default = a.is_default) for aidx, a in q.args.items()}
    if flipNeg and divideStr:
        new_semantics = 'logistic-x'
    elif flipNeg:
        new_semantics = 'logistic-1'
    elif divideStr:
        new_semantics = 'logistic-2'
    else:
        new_semantics = 'logistic'
    ans = QBAF(the_args, set(q.atts), set(q.sups), new_semantics)
    ans.rely_on = dict(q.rely_on)
    ans.out_degrees = dict(q.out_degrees)
    ans.final_strengths = dict(q.final_strengths)
    
    if flipNeg:
        # Step 1: Dealing with the negative final strengths
        # Modify arguments
        for aidx in ans.args:
            if ans.final_strengths[aidx] < 0: # flip
                ans.final_strengths[aidx] = -ans.final_strengths[aidx]
                ans.args[aidx].base_score = -ans.args[aidx].base_score
                if ans.args[aidx].support_class == 'Positive':
                    ans.args[aidx].support_class = 'Negative'
                elif ans.args[aidx].support_class == 'Negative':
                    ans.args[aidx].support_class = 'Positive'
                else:
                    assert False, "The original support class is neither Positive nor Negative."

        # Rewrite attacks, supports
        the_atts, the_sups = set(), set()
        for e in ans.rels:
            if ans.final_strengths[e[0]] == 0:
                continue
            if ans.args[e[0]].support_class != ans.args[e[1]].support_class:
                the_atts.add(e)
            else:
                the_sups.add(e)
        ans.atts, ans.sups = the_atts, the_sups

        ans.attacked, ans.supported = {aidx: set() for aidx in ans.args}, {aidx: set() for aidx in ans.args}
        for e in ans.atts:
            ans.attacked[e[1]].add(e[0])
        for e in ans.sups:
            ans.supported[e[1]].add(e[0])
    
    if divideStr:
        # Step 2: Dealing with the out_degrees
        if q.semantics == 'logistic':
            for aidx in ans.args:
                ans.final_strengths[aidx] /= max(1, ans.out_degrees[aidx])
                ans.args[aidx].base_score /= max(1, ans.out_degrees[aidx])

    return ans

# Generate plotly visualization
def get_level(idx, rels):
    if idx == 0:
        return 0
    else:
        above = [r[1] for r in rels if r[0] == idx]
        return max([get_level(a, rels) + 1 for a in above])

def generate_visualization(q: QBAF, 
                            matches: Dict[int,str],
                            color_pos: str = "green",
                            color_neg: str = "red",
                            color_sup: str = "black",
                            color_att: str = "orange",
                            show_fig: bool = True,
                            save_fig: bool = True, 
                            save_path: str = "test",
                            sort_by: str = "random"): # random or final_strength 
    sign = {'Positive': 1, 'Negative': -1}
    node_layout = {}
    for aidx, a in q.args.items():
        depth = get_level(aidx, q.rels)
        if depth not in node_layout:
            node_layout[depth] = []
        node_layout[depth].append(a)
    
    node_positions = {}
    ave_parents_pos = {}
    for k in range(max(node_layout)+1):
        if k > 0: # Sort the nodes of the same level based on the average location of their parents
            for a in node_layout[k]:
                above = [r[1] for r in q.rels if r[0] == a.id]
                ave_parents_pos[a.id] = np.mean([node_positions[b][0] for b in above])
            if sort_by == 'random':
                node_layout[k].sort(key=lambda a: (ave_parents_pos[a.id],random.random()))
            elif sort_by == 'final_strength':
                node_layout[k].sort(key=lambda a: (ave_parents_pos[a.id], sign[a.support_class] * q.final_strengths[a.id]))
            else:
                assert False, f"Unrecognized sort_by: {sort_by}"
        for idx, a in enumerate(node_layout[k]):
            node_positions[a.id] = (idx - (len(node_layout[k])-1)/2, -k)
        
    max_width = max([len(node_layout[k]) for k in node_layout])
    
    node_x = []
    node_y = []
    for aidx, pos in node_positions.items():
        node_x.append(pos[0])
        node_y.append(pos[1])
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            cmin = 0,
            cmax = 1,
            colorscale=[[0, color_pos], [0.5, '#CCCCCC'], [1.0, color_neg]],
            reversescale=True,
            color=[],
            size=20,
            colorbar=dict(
                thickness=15,
                title='Probability',
                xanchor='left',
                titleside='right'
            ),
            line_width=3))
    
    # node_trace.marker.size = [np.sqrt(p.coverage)*50 for p in the_list]
    node_trace.marker.color = [1 / (1 + np.exp(-q.final_strengths[aidx])) if q.args[aidx].support_class == 'Positive' else 1 / (1 + np.exp(q.final_strengths[aidx])) for aidx in node_positions]
    node_trace.marker.line.cmin = 0
    node_trace.marker.line.cmax = 1
    node_trace.marker.line.colorscale = [[0, color_pos], [0.5, '#CCCCCC'], [1.0, color_neg]]
    node_trace.marker.line.color = [1 / (1 + np.exp(q.args[aidx].base_score)) if q.args[aidx].support_class == 'Positive' else 1 / (1 + np.exp(-q.args[aidx].base_score)) for aidx in node_positions]
    
    txts = []
    for aidx in node_positions:
        a = q.args[aidx]
        s = f'#{aidx} {a.pattern.get_pattern_id()}<br>'
        if aidx != 0:
            s += f'Match: {matches[aidx]}<br>'
        s += f'Base score = {a.base_score:.3f}<br>'
        s += f'Final strength = {q.final_strengths[aidx]:.3f}<br>'
        s += f'Support class = {a.support_class}<br>'
        txts.append(s)
    node_trace.text = txts

    edge_x = []
    edge_y = []
    x0s = []
    y0s = []
    x1s = []
    y1s = []
    edge_colors = []
    for ridx, rtype in enumerate([q.atts, q.sups]):
        for edge in rtype:
            x0, y0 = node_positions[edge[0]]
            x1, y1 = node_positions[edge[1]]
            x0s.append(x0)
            y0s.append(y0)
            x1s.append(x1)
            y1s.append(y1)
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            c = color_sup if ridx else color_att  
            edge_colors.append(c)

    fig = go.Figure(data=[node_trace],
             layout=go.Layout(
#                 plot_bgcolor="#BBBBBB",
                title=f'Argumentation Graph',
                titlefont_size=14,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations = [dict(ax=x0s[i], ay=y0s[i], axref='x', ayref='y',
                        x=x1s[i], y=y1s[i], xref='x', yref='y', arrowwidth = 1,
                        showarrow=True, arrowhead=2, arrowcolor=edge_colors[i]) for i in range(0, len(x0s))],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    if save_fig:
        fig.write_html(f"{save_path}.html")
        fig.write_image(f"{save_path}.pdf")
        fig.write_image(f"{save_path}.png")

    if show_fig:
        fig.show()
    
    