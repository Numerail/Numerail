From Stdlib Require Import List.
From Stdlib Require Import Bool.
Import ListNotations.

Parameter vec : Type.

Inductive Result : Type :=
  | Approve : Result
  | Project : Result
  | Reject  : Result.

Record EnforcementOutput : Type := mk_output {
  result  : Result;
  enforced : vec;
}.

Record Constraint : Type := mk_constraint {
  is_satisfied : vec -> bool;
}.

Definition Region := list Constraint.

Fixpoint is_feasible (r : Region) (x : vec) : bool :=
  match r with
  | []      => true
  | c :: cs => is_satisfied c x && is_feasible cs x
  end.

Lemma is_feasible_correct :
  forall (r : Region) (x : vec),
    is_feasible r x = true <->
    (forall c, In c r -> is_satisfied c x = true).
Proof.
  induction r as [| c cs IH]; simpl; split; intros H.
  - intros c' Hin. contradiction.
  - reflexivity.
  - apply andb_true_iff in H. destruct H as [Hc Hcs].
    intros c' [Heq | Hin].
    + subst. exact Hc.
    + apply IH; assumption.
  - apply andb_true_iff. split.
    + apply H. left. reflexivity.
    + apply IH. intros c' Hin. apply H. right. exact Hin.
Qed.

Record ProjectionResult : Type := mk_proj {
  proj_point : vec;
  postcheck_passed : bool;
}.

Parameter solver : vec -> Region -> ProjectionResult.

Axiom project_postcheck :
  forall (x : vec) (r : Region),
    postcheck_passed (solver x r) = true ->
    is_feasible r (proj_point (solver x r)) = true.

Parameter operational_filters_pass : vec -> vec -> bool.

Record EnforcementConfig : Type := mk_config {
  has_hard_walls     : bool;
  hard_wall_violated : vec -> bool;
  mode_is_reject     : bool;
}.

Definition emit (res : Result) (y : vec) (r : Region)
  : option EnforcementOutput :=
  match res with
  | Approve => if is_feasible r y then Some (mk_output Approve y)
               else None
  | Project => if is_feasible r y then Some (mk_output Project y)
               else None
  | Reject  => Some (mk_output Reject y)
  end.

Lemma emit_path_invariant :
  forall (res : Result) (y : vec) (r : Region) (out : EnforcementOutput),
    emit res y r = Some out ->
    result out = Approve \/ result out = Project ->
    is_feasible r (enforced out) = true.
Proof.
  intros res y r out Hemit Hres.
  destruct res.
  - simpl in Hemit.
    destruct (is_feasible r y) eqn:Hf.
    + injection Hemit as Heq. subst out. simpl. exact Hf.
    + discriminate Hemit.
  - simpl in Hemit.
    destruct (is_feasible r y) eqn:Hf.
    + injection Hemit as Heq. subst out. simpl. exact Hf.
    + discriminate Hemit.
  - simpl in Hemit. injection Hemit as Heq. subst out.
    simpl in Hres. destruct Hres as [H | H]; discriminate H.
Qed.

Lemma reject_contradicts :
  forall (y : vec) (r : Region) (out : EnforcementOutput),
    emit Reject y r = Some out ->
    result out = Approve \/ result out = Project ->
    False.
Proof.
  intros y r out Hemit Hres.
  simpl in Hemit. injection Hemit as Heq. subst out.
  simpl in Hres. destruct Hres as [H | H]; discriminate H.
Qed.

Definition enforce (x : vec) (r : Region) (cfg : EnforcementConfig)
  : option EnforcementOutput :=
  if is_feasible r x then
    emit Approve x r
  else if has_hard_walls cfg && hard_wall_violated cfg x then
    emit Reject x r
  else if mode_is_reject cfg then
    emit Reject x r
  else
    let proj := solver x r in
    if postcheck_passed proj then
      if operational_filters_pass x (proj_point proj) then
        emit Project (proj_point proj) r
      else
        emit Reject x r
    else
      emit Reject x r.

Theorem enforcement_soundness :
  forall (x : vec) (r : Region) (cfg : EnforcementConfig)
         (out : EnforcementOutput),
    enforce x r cfg = Some out ->
    result out = Approve \/ result out = Project ->
    is_feasible r (enforced out) = true.
Proof.
  intros x r cfg out Henf Hres.
  unfold enforce in Henf.
  destruct (is_feasible r x) eqn:Hfeas.
  - eapply emit_path_invariant; eauto.
  - destruct (has_hard_walls cfg && hard_wall_violated cfg x) eqn:Hhw.
    + exfalso. eapply reject_contradicts; eauto.
    + destruct (mode_is_reject cfg) eqn:Hmode.
      * exfalso. eapply reject_contradicts; eauto.
      * remember (solver x r) as proj eqn:Hproj in Henf.
        destruct (postcheck_passed proj) eqn:Hpc.
        { destruct (operational_filters_pass x (proj_point proj)) eqn:Hops.
          - eapply emit_path_invariant; eauto.
          - exfalso. eapply reject_contradicts; eauto. }
        { exfalso. eapply reject_contradicts; eauto. }
Qed.

Corollary enforcement_soundness_per_constraint :
  forall (x : vec) (r : Region) (cfg : EnforcementConfig)
         (out : EnforcementOutput) (c : Constraint),
    enforce x r cfg = Some out ->
    result out = Approve \/ result out = Project ->
    In c r ->
    is_satisfied c (enforced out) = true.
Proof.
  intros x r cfg out c Henf Hres Hin.
  assert (Hfeas : is_feasible r (enforced out) = true)
    by (eapply enforcement_soundness; eauto).
  rewrite is_feasible_correct in Hfeas.
  apply Hfeas. exact Hin.
Qed.

Theorem fail_closed :
  forall (x : vec) (r : Region) (cfg : EnforcementConfig)
         (out : EnforcementOutput),
    is_feasible r x = false ->
    has_hard_walls cfg && hard_wall_violated cfg x = false ->
    mode_is_reject cfg = false ->
    postcheck_passed (solver x r) = false ->
    enforce x r cfg = Some out ->
    result out = Reject.
Proof.
  intros x r cfg out Hinfeas Hhw Hmode Hpc Henf.
  unfold enforce in Henf.
  rewrite Hinfeas in Henf.
  rewrite Hhw in Henf.
  rewrite Hmode in Henf.
  simpl in Henf.
  rewrite Hpc in Henf.
  simpl in Henf.
  injection Henf as Heq. subst out. reflexivity.
Qed.

Theorem hard_wall_dominance :
  forall (x : vec) (r : Region) (cfg : EnforcementConfig)
         (out : EnforcementOutput),
    is_feasible r x = false ->
    has_hard_walls cfg = true ->
    hard_wall_violated cfg x = true ->
    enforce x r cfg = Some out ->
    result out = Reject.
Proof.
  intros x r cfg out Hinfeas Hhw_on Hhw_viol Henf.
  unfold enforce in Henf.
  rewrite Hinfeas in Henf.
  rewrite Hhw_on in Henf. rewrite Hhw_viol in Henf.
  simpl in Henf.
  injection Hemit as Heq. subst out. reflexivity.
Qed.

Theorem passthrough :
  forall (x : vec) (r : Region) (cfg : EnforcementConfig),
    is_feasible r x = true ->
    enforce x r cfg = Some (mk_output Approve x).
Proof.
  intros x r cfg Hfeas.
  unfold enforce. rewrite Hfeas.
  unfold emit. rewrite Hfeas. reflexivity.
Qed.

Theorem idempotence :
  forall (x : vec) (r : Region) (cfg1 cfg2 : EnforcementConfig)
         (out : EnforcementOutput),
    enforce x r cfg1 = Some out ->
    result out = Approve \/ result out = Project ->
    enforce (enforced out) r cfg2 =
      Some (mk_output Approve (enforced out)).
Proof.
  intros x r cfg1 cfg2 out Henf Hres.
  apply passthrough.
  eapply enforcement_soundness; eauto.
Qed.

Lemma budget_monotonicity :
  forall (c_tight c_orig : Constraint) (r_rest : Region) (x : vec),
    (forall z, is_satisfied c_tight z = true ->
               is_satisfied c_orig z = true) ->
    is_feasible (c_tight :: r_rest) x = true ->
    is_feasible (c_orig :: r_rest) x = true.
Proof.
  intros c_tight c_orig r_rest x Htight Hfeas.
  simpl in *. apply andb_true_iff in Hfeas.
  destruct Hfeas as [Hc Hrs].
  apply andb_true_iff. split.
  - apply Htight. exact Hc.
  - exact Hrs.
Qed.

Theorem numerail_guarantee :
  forall (x : vec) (r : Region) (cfg : EnforcementConfig)
         (out : EnforcementOutput),
    enforce x r cfg = Some out ->
    result out = Approve \/ result out = Project ->
    forall c, In c r -> is_satisfied c (enforced out) = true.
Proof.
  intros x r cfg out Henf Hres c Hin.
  eapply enforcement_soundness_per_constraint; eauto.
Qed.
