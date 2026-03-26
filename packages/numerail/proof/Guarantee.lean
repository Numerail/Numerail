import Mathlib.Tactic

axiom Vec : Type

inductive Result where
  | approve : Result
  | project : Result
  | reject  : Result
deriving DecidableEq

structure EnforcementOutput where
  result   : Result
  enforced : Vec

structure Constraint where
  is_satisfied : Vec → Bool

abbrev Region := List Constraint

def is_feasible (r : Region) (x : Vec) : Bool :=
  r.all (fun c => c.is_satisfied x)

theorem is_feasible_correct (r : Region) (x : Vec) :
    is_feasible r x = true ↔
    ∀ c, c ∈ r → c.is_satisfied x = true := by
  induction r with
  | nil => simp [is_feasible, List.all]
  | cons c cs ih => simp [is_feasible, List.all, Bool.and_eq_true]

structure ProjectionResult where
  proj_point       : Vec
  postcheck_passed : Bool

noncomputable axiom solver : Vec → Region → ProjectionResult

axiom project_postcheck :
  ∀ (x : Vec) (r : Region),
    (solver x r).postcheck_passed = true →
    is_feasible r (solver x r).proj_point = true

noncomputable axiom operational_filters_pass : Vec → Vec → Bool

structure EnforcementConfig where
  has_hard_walls     : Bool
  hard_wall_violated : Vec → Bool
  mode_is_reject     : Bool

def emit (res : Result) (y : Vec) (r : Region)
    : Option EnforcementOutput :=
  match res with
  | .approve => if is_feasible r y then
                  some ⟨.approve, y⟩
                else none
  | .project => if is_feasible r y then
                  some ⟨.project, y⟩
                else none
  | .reject  => some ⟨.reject, y⟩

private lemma approve_sound (y : Vec) (r : Region) (out : EnforcementOutput)
    (h : emit .approve y r = some out) :
    is_feasible r out.enforced = true := by
  unfold emit at h
  split_ifs at h with hf
  · injection h with h'; subst h'; exact hf

private lemma project_sound (y : Vec) (r : Region) (out : EnforcementOutput)
    (h : emit .project y r = some out) :
    is_feasible r out.enforced = true := by
  unfold emit at h
  split_ifs at h with hf
  · injection h with h'; subst h'; exact hf

private lemma reject_absurd (y : Vec) (r : Region)
    (out : EnforcementOutput)
    (h_emit : emit .reject y r = some out)
    (h_res : out.result = .approve ∨ out.result = .project) : False := by
  unfold emit at h_emit
  injection h_emit with h'
  subst h'
  rcases h_res with h | h <;> simp at h

noncomputable def enforce (x : Vec) (r : Region) (cfg : EnforcementConfig)
    : Option EnforcementOutput :=
  if is_feasible r x then
    emit .approve x r
  else if cfg.has_hard_walls && cfg.hard_wall_violated x then
    emit .reject x r
  else if cfg.mode_is_reject then
    emit .reject x r
  else if (solver x r).postcheck_passed then
    if operational_filters_pass x (solver x r).proj_point then
      emit .project (solver x r).proj_point r
    else
      emit .reject x r
  else
    emit .reject x r

theorem enforcement_soundness
    (x : Vec) (r : Region) (cfg : EnforcementConfig)
    (out : EnforcementOutput)
    (h_enf : enforce x r cfg = some out)
    (h_res : out.result = .approve ∨ out.result = .project) :
    is_feasible r out.enforced = true := by
  unfold enforce at h_enf
  split_ifs at h_enf
  · exact approve_sound _ r out h_enf
  · exact absurd h_res (reject_absurd _ r out h_enf)
  · exact absurd h_res (reject_absurd _ r out h_enf)
  · exact project_sound _ r out h_enf
  · exact absurd h_res (reject_absurd _ r out h_enf)
  · exact absurd h_res (reject_absurd _ r out h_enf)

theorem enforcement_soundness_per_constraint
    (x : Vec) (r : Region) (cfg : EnforcementConfig)
    (out : EnforcementOutput) (c : Constraint)
    (h_enf : enforce x r cfg = some out)
    (h_res : out.result = .approve ∨ out.result = .project)
    (h_in : c ∈ r) :
    c.is_satisfied out.enforced = true := by
  have hf := enforcement_soundness x r cfg out h_enf h_res
  exact (is_feasible_correct r out.enforced).mp hf c h_in

theorem fail_closed
    (x : Vec) (r : Region) (cfg : EnforcementConfig)
    (out : EnforcementOutput)
    (h_infeas : is_feasible r x = false)
    (h_hw : (cfg.has_hard_walls && cfg.hard_wall_violated x) = false)
    (h_mode : cfg.mode_is_reject = false)
    (h_pc : (solver x r).postcheck_passed = false)
    (h_enf : enforce x r cfg = some out) :
    out.result = .reject := by
  unfold enforce at h_enf
  simp [h_infeas, h_hw, h_mode, h_pc, emit] at h_enf
  subst h_enf
  rfl

theorem hard_wall_dominance
    (x : Vec) (r : Region) (cfg : EnforcementConfig)
    (out : EnforcementOutput)
    (h_infeas : is_feasible r x = false)
    (h_hw_on : cfg.has_hard_walls = true)
    (h_hw_viol : cfg.hard_wall_violated x = true)
    (h_enf : enforce x r cfg = some out) :
    out.result = .reject := by
  unfold enforce at h_enf
  simp [h_infeas, h_hw_on, h_hw_viol, emit] at h_enf
  subst h_enf
  rfl

theorem passthrough
    (x : Vec) (r : Region) (cfg : EnforcementConfig)
    (h_feas : is_feasible r x = true) :
    enforce x r cfg = some ⟨.approve, x⟩ := by
  unfold enforce
  simp [h_feas, emit]

theorem idempotence
    (x : Vec) (r : Region) (cfg1 cfg2 : EnforcementConfig)
    (out : EnforcementOutput)
    (h_enf : enforce x r cfg1 = some out)
    (h_res : out.result = .approve ∨ out.result = .project) :
    enforce out.enforced r cfg2 = some ⟨.approve, out.enforced⟩ := by
  exact passthrough out.enforced r cfg2
    (enforcement_soundness x r cfg1 out h_enf h_res)

lemma budget_monotonicity
    (c_tight c_orig : Constraint) (r_rest : Region) (x : Vec)
    (h_tight : ∀ z, c_tight.is_satisfied z = true →
                     c_orig.is_satisfied z = true)
    (h_feas : is_feasible (c_tight :: r_rest) x = true) :
    is_feasible (c_orig :: r_rest) x = true := by
  simp [is_feasible, Bool.and_eq_true] at *
  exact ⟨h_tight x h_feas.1, h_feas.2⟩

theorem numerail_guarantee
    (x : Vec) (r : Region) (cfg : EnforcementConfig)
    (out : EnforcementOutput)
    (h_enf : enforce x r cfg = some out)
    (h_res : out.result = .approve ∨ out.result = .project) :
    ∀ c, c ∈ r → c.is_satisfied out.enforced = true := by
  intro c h_in
  exact enforcement_soundness_per_constraint x r cfg out c h_enf h_res h_in
