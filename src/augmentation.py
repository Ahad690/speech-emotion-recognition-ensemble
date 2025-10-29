def evaluate_with_tta(model, test_loader, device, n_augmentations=5):
    """
    Evaluate with Test Time Augmentation
    This can improve accuracy by 0.5-2% without retraining!
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs_list = []
    
    print(f"Applying Test Time Augmentation with {n_augmentations} augmentations per sample...")
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="TTA Testing"):
            batch_probs = []
            
            # Original prediction
            features_gpu = features.to(device)
            with autocast():
                outputs = model(features_gpu)
            probs = F.softmax(outputs, dim=1)
            batch_probs.append(probs.cpu())
            
            # Augmented predictions
            for aug_idx in range(n_augmentations - 1):
                # Apply different augmentations
                aug_features = features.clone()
                
                if aug_idx == 0:
                    # Add slight noise
                    aug_features = aug_features + torch.randn_like(aug_features) * 0.003
                elif aug_idx == 1:
                    # Slight time shift
                    shift_amount = torch.randint(-5, 5, (1,)).item()
                    aug_features = torch.roll(aug_features, shifts=shift_amount, dims=-1)
                elif aug_idx == 2:
                    # Slight amplitude scaling
                    scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1
                    aug_features = aug_features * scale
                else:
                    # Random small perturbation
                    aug_features = aug_features + torch.randn_like(aug_features) * 0.002
                
                aug_features = aug_features.to(device)
                with autocast():
                    aug_outputs = model(aug_features)
                aug_probs = F.softmax(aug_outputs, dim=1)
                batch_probs.append(aug_probs.cpu())
            
            # Average predictions from all augmentations
            avg_probs = torch.stack(batch_probs).mean(dim=0)
            _, predicted = avg_probs.max(1)
            
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
            all_probs_list.append(avg_probs.numpy())
    
    return np.array(all_preds), np.array(all_labels), np.vstack(all_probs_list)

# Apply TTA to your already trained model
print("="*60)
print("EVALUATING WITH TEST TIME AUGMENTATION")
print("="*60)

# Load best model if not already loaded
if not 'model' in globals():
    checkpoint = torch.load('/kaggle/working/best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model with {checkpoint['val_acc']:.2f}% validation accuracy")

# Run TTA evaluation
tta_preds, tta_labels, tta_probs = evaluate_with_tta(model, test_loader, device, n_augmentations=5)

# Calculate improved metrics
tta_accuracy = accuracy_score(tta_labels, tta_preds)
print(f"\n🎯 Original Test Accuracy: 71.93%")
print(f"🚀 TTA Test Accuracy: {tta_accuracy:.4f} ({tta_accuracy*100:.2f}%)")
print(f"📈 Improvement: +{(tta_accuracy - 0.7193)*100:.2f}%")

# Detailed classification report
print("\n" + "="*60)
print("TTA Classification Report:")
print("="*60)
print(classification_report(tta_labels, tta_preds, target_names=emotion_names[:config.n_classes], digits=3))

# Confusion Matrix
tta_cm = confusion_matrix(tta_labels, tta_preds)
print("\n" + "="*60)
print("TTA Confusion Matrix:")
print("="*60)
print(tta_cm)

# Per-class accuracy
tta_per_class_acc = tta_cm.diagonal() / tta_cm.sum(axis=1)
print("\n" + "="*60)
print("TTA Per-Class Accuracy:")
print("="*60)
for i, emotion in enumerate(emotion_names[:config.n_classes]):
    if i < len(tta_per_class_acc):
        improvement = (tta_per_class_acc[i] - per_class_acc[i]) * 100
        print(f"  {emotion:10s}: {tta_per_class_acc[i]:.3f} ({tta_per_class_acc[i]*100:.1f}%) [{'↑' if improvement > 0 else '↓'}{abs(improvement):.1f}%]")

print("\n✅ TTA Evaluation Complete!")
print("💡 TTA typically improves accuracy by 0.5-2% without any retraining!")